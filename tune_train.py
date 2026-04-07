"""
tune.py — Optuna-based hyperparameter tuning for NeuralDecisionForest.

Data splits
-----------
  train  → used to fit the model inside every trial and the final retraining
  val    → used for early stopping + Optuna objective (never seen by Optuna sampler directly)
  test   → touched exactly once, after the final retrained model is ready

Run (monitor validation loss):
    python tune.py -dataset mnist -n_trials 50 -epochs 30 -es_monitor val_loss -gpuid 0

Run (monitor validation accuracy):
    python tune.py -dataset mnist -n_trials 50 -epochs 30 -es_monitor val_acc -gpuid 0
"""

import argparse
import os
import logging
import tempfile
import optuna
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split, ConcatDataset
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from tqdm import tqdm

import dataset
import ndf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arg():
    parser = argparse.ArgumentParser(description='tune.py — NDF hyperparameter search and training...')
    parser.add_argument('-dataset',    choices=['mnist', 'adult', 'letter', 'yeast'], default='mnist')
    parser.add_argument('-n_trials',   type=int,   default=50)
    parser.add_argument('-epochs',     type=int,   default=30,  help='Max epochs per trial / final training')
    parser.add_argument('-gpuid',      type=int,   default=-1)
    parser.add_argument('-n_class',    type=int,   default=10)
    parser.add_argument('-study_name', type=str,   default='ndf_study')
    parser.add_argument('-storage',    type=str,   default=None,
                        help='Optional Optuna DB URL, e.g. sqlite:///ndf.db')
    # Early-stopping knobs
    parser.add_argument('-es_monitor',   choices=['val_loss', 'val_acc'], default='val_loss',
                        help='Metric to monitor for early stopping. '
                             '"val_loss" (lower-is-better) or "val_acc" (higher-is-better).')
    parser.add_argument('-es_patience',  type=int,   default=5,
                        help='Epochs without improvement before stopping')
    parser.add_argument('-es_min_delta', type=float, default=1e-4,
                        help='Minimum change to count as an improvement')
    # Val / test fractions (used for datasets without a canonical test split)
    parser.add_argument('-val_fraction',  type=float, default=0.15)
    parser.add_argument('-test_fraction', type=float, default=0.15)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stops training when a monitored metric stops improving.

    Parameters
    ----------
    mode      : 'min' for loss-like metrics (lower is better),
                'max' for accuracy-like metrics (higher is better).
    patience  : epochs to wait after the last improvement before stopping.
    min_delta : minimum change in the monitored metric to count as an
                improvement. Applied in the correct direction for each mode:
                  'min' → new_val < best - min_delta
                  'max' → new_val > best + min_delta

    Usage
    -----
        es = EarlyStopping(mode='min', patience=5)
        for epoch in ...:
            metrics = evaluate(model, val_loader, device)
            es.step(metrics['val_loss'], model)
            if es.should_stop:
                break
        es.restore_best(model)   # roll back to the best-epoch weights
    """

    def __init__(self, mode: str = 'min', patience: int = 5, min_delta: float = 1e-4):
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.mode         = mode
        self.patience     = patience
        self.min_delta    = min_delta
        self.best_value   = float('inf') if mode == 'min' else -float('inf')
        self.best_weights = None          # deep-copy of state_dict at best epoch (memory-intensive)
        # self.best_ckpt_path = tempfile.NamedTemporaryFile(delete=False, dir='./').name # (memory-friendly)

        fd, self._ckpt_path = tempfile.mkstemp(suffix='.pt', prefix='es_ckpt_', dir='./',)
        os.close(fd)

        self.counter      = 0
        self.should_stop  = False

    def _is_improvement(self, new_value: float) -> bool:
        if self.mode == 'min':
            return new_value < self.best_value - self.min_delta
        else:  # 'max'
            return new_value > self.best_value + self.min_delta

    def step(self, metric_value: float, model: torch.nn.Module) -> None:
        """
        Call once per epoch with the current value of the monitored metric.
        Automatically snapshots the model weights on every improvement.
        """
        if self._is_improvement(metric_value):
            self.best_value   = metric_value
            # self.best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), self._ckpt_path)  # save to disk
            self.counter      = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: torch.nn.Module) -> None:
        """Load the weights from the best epoch back into the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
        device = next(model.parameters()).device
        model.load_state_dict(
            torch.load(self._ckpt_path, map_location=device, weights_only=True)
        )

    def cleanup(self) -> None:
        """Delete the temporary checkpoint file from disk."""
        try:
            os.remove(self._ckpt_path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def prepare_db(opt) -> dict:
    """
    Return three non-overlapping splits: train / val / test.

    MNIST  : canonical 60k/10k split is kept for test; the 60k training
             portion is further divided into train / val.
    UCI    : no canonical split exists, so the full data is divided into
             train / val / test proportionally.
    """
    if opt.dataset == 'mnist':
        tfm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        full_train = torchvision.datasets.MNIST('./data/mnist', train=True,  download=True, transform=tfm)
        test_set   = torchvision.datasets.MNIST('./data/mnist', train=False, download=True, transform=tfm)

        n_val   = int(len(full_train) * opt.val_fraction)
        n_train = len(full_train) - n_val
        train_set, val_set = random_split(
            full_train, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

    else:
        loaders = {
            'adult':  lambda tr: dataset.UCIAdult('./data/uci_adult',   train=tr),
            'letter': lambda tr: dataset.UCILetter('./data/uci_letter', train=tr),
            'yeast':  lambda tr: dataset.UCIYeast('./data/uci_yeast',   train=tr),
        }
        full_data = ConcatDataset([loaders[opt.dataset](True), loaders[opt.dataset](False)])
        n_total  = len(full_data)
        n_test   = int(n_total * opt.test_fraction)
        n_val    = int(n_total * opt.val_fraction)
        n_train  = n_total - n_val - n_test
        train_set, val_set, test_set = random_split(
            full_data, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )

    return {'train': train_set, 'val': val_set, 'test': test_set}


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def sample_hyperparameters(trial: optuna.Trial) -> dict:
    """All tuneable knobs in one place — add / remove freely."""
    return {
        # Forest architecture
        'n_tree':            trial.suggest_int  ('n_tree',            2,   80),
        'tree_depth':        trial.suggest_int  ('tree_depth',        2,    10),
        'tree_feature_rate': trial.suggest_float('tree_feature_rate', 0.1,  1.0),
        'jointly_training':  trial.suggest_categorical('jointly_training', [True, False]),

        # Feature extractor
        'feat_dropout':      trial.suggest_float('feat_dropout',      0.0,  0.5),
        # 'shallow':           trial.suggest_categorical('shallow', [True, False]),

        # Optimiser
        'lr':                trial.suggest_float('lr',           1e-4, 1e-1, log=True),
        'weight_decay':      trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        # 'batch_size':        trial.suggest_categorical('batch_size', [64, 128, 256]),
    }


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(hp: dict, dataset_name: str, n_class: int, device: torch.device) -> torch.nn.Module:
    # shallow = hp['shallow'] if dataset_name == 'mnist' else True
    shallow = False if dataset_name == 'mnist' else True

    feat_layer_cls = {
        'mnist':  ndf.MNISTFeatureLayer,
        'adult':  ndf.UCIAdultFeatureLayer,
        'letter': ndf.UCILetterFeatureLayer,
        'yeast':  ndf.UCIYeastFeatureLayer,
    }[dataset_name]

    feat_layer = feat_layer_cls(hp['feat_dropout'], shallow=shallow)
    forest = ndf.Forest(
        n_tree=hp['n_tree'],
        tree_depth=hp['tree_depth'],
        n_in_feature=feat_layer.get_out_feature_size(),
        tree_feature_rate=hp['tree_feature_rate'],
        n_class=n_class,
        jointly_training=hp['jointly_training'],
    )
    return ndf.NeuralDecisionForest(feat_layer, forest).to(device)


# ---------------------------------------------------------------------------
# Shared train / eval helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optim, device, jointly_training) -> None:

    if not jointly_training:
        # print("Epoch %d : Two Stage Learning - Update PI" % epoch)

        cls_onehot = torch.eye(model.forest.trees[0].n_class, device=device)
        feat_batches = []
        target_batches = []
        # train_loader = torch.utils.data.DataLoader(db['train'], batch_size=opt.batch_size, shuffle=True)

        with torch.no_grad():
            for data, target in loader:
                # CHANGED: use .to(device) throughout instead of .cuda() conditionals.
                data = data.to(device)
                target = target.to(device)
                # CHANGED: removed Variable() wrappers — fully deprecated since PyTorch 0.4.
                feats = model.feature_layer(data)
                feats = feats.view(feats.size(0), -1)
                feat_batches.append(feats)
                target_batches.append(cls_onehot[target])

            for tree in tqdm(model.forest.trees, desc='Updating...'):
                mu_batches = []
                for feats in feat_batches:
                    mu = tree(feats)  # [batch_size, n_leaf]
                    mu_batches.append(mu)

                for _ in range(20):
                    new_pi = torch.zeros((tree.n_leaf, tree.n_class), device=device)
                    for mu, target in zip(mu_batches, target_batches):
                        pi = tree.get_pi()                                          # [n_leaf, n_class]
                        prob = tree.cal_prob(mu, pi)                                # [batch_size, n_class]

                        # CHANGED: removed .data access — gradients are already
                        # disabled by the enclosing torch.no_grad() context.
                        _target = target.unsqueeze(1)                               # [batch_size, 1, n_class]
                        _pi = pi.unsqueeze(0)                                       # [1, n_leaf, n_class]
                        _mu = mu.unsqueeze(2)                                       # [batch_size, n_leaf, 1]
                        _prob = torch.clamp(prob.unsqueeze(1), min=1e-6, max=1.)    # [batch_size, 1, n_class]

                        _new_pi = torch.mul(torch.mul(_target, _pi), _mu) / _prob  # [batch_size, n_leaf, n_class]
                        new_pi += torch.sum(_new_pi, dim=0)

                    # CHANGED: removed Variable() wrapper around new_pi before softmax.
                    new_pi = F.softmax(new_pi, dim=1)
                    tree.update_pi(new_pi)

    # Update \Theta
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optim.zero_grad()
        F.nll_loss(torch.log(model(data)), target).backward()
        optim.step()


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """
    Return both validation loss and accuracy in a single pass.

    Returns
    -------
    dict with keys:
        'val_loss' : float  — mean NLL loss over the dataset (lower is better)
        'val_acc'  : float  — fraction of correctly classified samples (higher is better)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output      = model(data)
        total_loss += F.nll_loss(torch.log(output), target, reduction='sum').item()
        correct    += output.argmax(dim=1).eq(target).sum().item()
        total      += target.size(0)
    return {'val_loss': total_loss / total, 'val_acc': correct / total}


# ---------------------------------------------------------------------------
# run_trial — called once per Optuna trial
# ---------------------------------------------------------------------------

def run_trial(trial: optuna.Trial, db: dict, opt, device: torch.device) -> float:
    """
    Train one hyperparameter configuration for up to opt.epochs epochs.

    Two complementary stopping mechanisms run in parallel:

    1. EarlyStopping — monitors opt.es_monitor ('val_loss' or 'val_acc'),
       saves the best checkpoint, and halts training when the metric has not
       improved for opt.es_patience epochs.

    2. Optuna MedianPruner — independently prunes trials whose intermediate
       val_acc falls below the median of completed trials at the same epoch.
       Always uses val_acc as the Optuna objective regardless of es_monitor,
       because Optuna optimises a single direction (maximise accuracy).

    Returns best val_acc (Optuna objective), taken at the best-ES-checkpoint
    epoch rather than the final epoch.
    """
    hp    = sample_hyperparameters(trial)
    model = build_model(hp, opt.dataset, opt.n_class, device)
    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=hp['lr'], weight_decay=hp['weight_decay'],
    )

    train_loader = DataLoader(db['train'], batch_size=256,
                              shuffle=True,  # num_workers=2, pin_memory=True
                              )
    val_loader   = DataLoader(db['val'],   batch_size=256,
                              shuffle=False, # num_workers=2, pin_memory=True
                              )

    # mode='min' for val_loss, mode='max' for val_acc
    es_mode = 'min' if opt.es_monitor == 'val_loss' else 'max'
    es = EarlyStopping(mode=es_mode, patience=opt.es_patience, min_delta=opt.es_min_delta)

    try:
        for epoch in range(1, opt.epochs + 1):
            train_one_epoch(model, train_loader, optim, device, hp['jointly_training'])
            metrics = evaluate(model, val_loader, device)

            # 1. Early stopping — use whichever metric was requested
            es.step(metrics[opt.es_monitor], model)

            # 2. Optuna pruning — always reports val_acc (study direction = 'maximize')
            trial.report(metrics['val_acc'], epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if es.should_stop:
                logging.info(
                    f"Trial {trial.number}: early stopping at epoch {epoch} "
                    f"(best {opt.es_monitor} = {es.best_value:.4f})"
                )
                break

        # Restore the best-epoch weights; return the val_acc at that checkpoint
        es.restore_best(model)
        # Re-evaluate at the restored checkpoint so the returned score is consistent
        metrics = evaluate(model, val_loader, device)
        return metrics['val_acc']
    finally:
        # Always clean up the temp checkpoint file, even if the trial crashes
        # or is pruned — this prevents disk/handle leaks.
        es.cleanup()


# ---------------------------------------------------------------------------
# Final training — runs once after tuning completes
# ---------------------------------------------------------------------------

def final_training(best_params: dict, db: dict, opt, device: torch.device) -> float:
    """
    Retrain from scratch with the best hyperparameters on train ∪ val.

    Training data : train ∪ val  (maximises use of labelled data post-tuning)
    ES monitoring : val split     (still used for checkpoint selection)
    Test data     : test          (opened here for the first and only time)

    Returns final test accuracy.
    """
    print("\n" + "=" * 60)
    print(f"Final retraining  |  early stopping on: {opt.es_monitor}")
    print("=" * 60)

    model = build_model(best_params, opt.dataset, opt.n_class, device)
    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=best_params['lr'], weight_decay=best_params['weight_decay'],
    )

    train_loader = DataLoader(
        ConcatDataset([db['train'], db['val']]),
        batch_size=256, shuffle=True, 
    )
    val_loader  = DataLoader(db['val'],  batch_size=256,
                             shuffle=False, # num_workers=2, pin_memory=True
                             )
    test_loader = DataLoader(db['test'], batch_size=256,
                             shuffle=False, # num_workers=2, pin_memory=True
                             )

    es_mode = 'min' if opt.es_monitor == 'val_loss' else 'max'
    es = EarlyStopping(mode=es_mode, patience=opt.es_patience, min_delta=opt.es_min_delta)

    try:
        for epoch in range(1, opt.epochs + 1):
            train_one_epoch(model, train_loader, optim, device, best_params['jointly_training'])
            metrics = evaluate(model, val_loader, device)
            es.step(metrics[opt.es_monitor], model)

            # Show both metrics every epoch regardless of which one drives ES
            improved = es.counter == 0
            print(
                f"  Epoch {epoch:>3}/{opt.epochs}"
                f"  |  val_loss = {metrics['val_loss']:.4f}"
                f"  |  val_acc  = {metrics['val_acc']:.4f}"
                f"{'  ← best' if improved else f'  (no improvement {es.counter}/{es.patience})'}"
            )

            if es.should_stop:
                print(f"\n  Early stopping triggered at epoch {epoch}.")
                break

        es.restore_best(model)
        # Report final val metrics at the restored checkpoint
        val_metrics = evaluate(model, val_loader, device)
        print(f"\n  Best checkpoint — val_loss: {val_metrics['val_loss']:.4f}"
              f"  |  val_acc: {val_metrics['val_acc']:.4f}")

        # ---- held-out test evaluation (first and only time) ----
        test_metrics = evaluate(model, test_loader, device)
        print(f"  Test set        — test_loss: {test_metrics['val_loss']:.4f}"
              f"  |  test_acc: {test_metrics['val_acc']:.4f}")
        print("=" * 60)

        # Persist the final model alongside its hyperparameters
        save_path = f"{opt.study_name}_best_model.pt"
        torch.save({'model_state_dict': model.state_dict(), 'hyperparameters': best_params}, save_path)
        print(f"\nFinal model saved to: {save_path}")

        return test_metrics['val_acc']
    finally:
        es.cleanup()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    opt = parse_arg()

    device = (
        torch.device('cuda', opt.gpuid)
        if opt.gpuid >= 0 and torch.cuda.is_available()
        else torch.device('cpu')
    )
    print(f"Device       : {device}")
    print(f"ES monitor   : {opt.es_monitor}  "
          f"(patience={opt.es_patience}, min_delta={opt.es_min_delta})")

    db = prepare_db(opt)
    print(f"Split sizes  — train: {len(db['train'])}  "
          f"val: {len(db['val'])}  test: {len(db['test'])}")

    # ---- Optuna study ----
    study = optuna.create_study(
        study_name=opt.study_name,
        direction='maximize',        # Optuna always maximises val_acc
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        storage=opt.storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: run_trial(trial, db, opt, device),
        n_trials=opt.n_trials,
        show_progress_bar=True,
    )

    # ---- Tuning summary ----
    print("\n" + "=" * 60)
    print("Hyperparameter search complete.")
    print(f"  Best val accuracy (tuning) : {study.best_value:.4f}")
    print("  Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k:<22} = {v}")

    csv_path = f"{opt.study_name}_trials.csv"
    study.trials_dataframe().to_csv(csv_path, index=False)
    print(f"\nFull trial log saved to: {csv_path}")

    # ---- Final training + test evaluation ----
    test_acc = final_training(study.best_params, db, opt, device)
    print(f"\nFinal held-out test accuracy: {test_acc:.4f}")

    # ---- Visualisations (optional, requires plotly) ----
    try:
        import optuna.visualization as vis
        for fig, name in [
            (vis.plot_optimization_history(study), 'history'),
            (vis.plot_param_importances(study),    'importances'),
            (vis.plot_parallel_coordinate(study),  'parallel'),
        ]:
            path = f"{opt.study_name}_{name}.html"
            fig.write_html(path)
            print(f"Visualisation saved: {path}")
    except ImportError:
        print("Install plotly for visualisations:  pip install optuna[visualization]")


if __name__ == '__main__':
    main()
