"""
tune_train.py — Optuna hyperparameter search + final training for all NDF variants.

Variants
--------
  leaf_dist   — original class-distribution leaves (jointly or two-stage)
  leaf_proto  — latent prototype leaves + shared task head
  leaf_expert — local linear expert leaves

Data splits
-----------
  train  → used to fit the model inside every trial and the final retraining
  val    → used for early stopping + Optuna objective
  test   → touched exactly once, after the final retrained model is ready

Usage examples
--------------
  python tune_train.py -dataset mnist -variant leaf_dist  -n_trials 50 -epochs 30
  python tune_train.py -dataset mnist -variant leaf_proto  -n_trials 50 -epochs 30
  python tune_train.py -dataset mnist -variant leaf_expert -n_trials 50 -epochs 30

The saved checkpoint includes variant metadata so transfer.py can load it:
  torch.load('{study_name}_best_model.pt') →
      {'model_state_dict', 'hyperparameters', 'variant', 'n_class', 'dataset'}
"""

import argparse
import logging
import os
import tempfile

import optuna
import torch
import torch.nn.functional as F
import torchvision
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm import tqdm

import dataset
import ndf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arg():
    parser = argparse.ArgumentParser(
        description='tune_train.py — NDF hyperparameter search and training')
    parser.add_argument('-dataset',    choices=['mnist', 'adult', 'letter', 'yeast'], default='mnist')
    parser.add_argument('-n_trials',   type=int,   default=50)
    parser.add_argument('-epochs',     type=int,   default=30,
                        help='Max epochs per trial / final training')
    parser.add_argument('-gpuid',      type=int,   default=-1)
    parser.add_argument('-n_class',    type=int,   default=10)
    parser.add_argument('-study_name', type=str,   default='ndf_study')
    parser.add_argument('-storage',    type=str,   default=None,
                        help='Optional Optuna DB URL, e.g. sqlite:///ndf.db')

    # --- variant ---
    parser.add_argument('-variant', choices=['leaf_dist', 'leaf_proto', 'leaf_expert'],
                        default='leaf_dist',
                        help='Leaf parameterisation: leaf_dist (original) | '
                             'leaf_proto (prototype vectors) | leaf_expert (local linear models)')

    # --- early stopping ---
    parser.add_argument('-es_monitor',   choices=['val_loss', 'val_acc'], default='val_loss')
    parser.add_argument('-es_patience',  type=int,   default=5)
    parser.add_argument('-es_min_delta', type=float, default=1e-4)

    # --- data splits ---
    parser.add_argument('-val_fraction',  type=float, default=0.15)
    parser.add_argument('-test_fraction', type=float, default=0.15)

    # --- output ---
    parser.add_argument('-save_path', type=str, default=None,
                        help='Override the checkpoint save path '
                             '(default: {study_name}_best_model.pt)')

    opt = parser.parse_args()
    if opt.save_path is None:
        opt.save_path = f"{opt.study_name}_best_model.pt"
    return opt


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stops training when a monitored metric stops improving.
    Checkpoints are written to disk to avoid holding a full copy in RAM.

    Parameters
    ----------
    mode      : 'min' (lower-is-better) or 'max' (higher-is-better).
    patience  : epochs to wait after the last improvement before stopping.
    min_delta : minimum change to qualify as an improvement.
    """

    def __init__(self, mode: str = 'min', patience: int = 5, min_delta: float = 1e-4):
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.mode        = mode
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_value  = float('inf') if mode == 'min' else -float('inf')
        self.counter     = 0
        self.should_stop = False
        self._ckpt_saved = False

        fd, self._ckpt_path = tempfile.mkstemp(suffix='.pt', prefix='es_ckpt_', dir='./')
        os.close(fd)

    def _is_improvement(self, v: float) -> bool:
        return (v < self.best_value - self.min_delta if self.mode == 'min'
                else v > self.best_value + self.min_delta)

    def step(self, metric_value: float, model: torch.nn.Module) -> None:
        if self._is_improvement(metric_value):
            self.best_value  = metric_value
            torch.save(model.state_dict(), self._ckpt_path)
            self._ckpt_saved = True
            self.counter     = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: torch.nn.Module) -> None:
        if not self._ckpt_saved:
            logging.warning("EarlyStopping.restore_best(): no checkpoint was saved.")
            return
        device = next(model.parameters()).device
        model.load_state_dict(
            torch.load(self._ckpt_path, map_location=device, weights_only=True)
        )

    def cleanup(self) -> None:
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

    MNIST  : canonical 60k/10k split kept for test; 60k split into train/val.
    UCI    : full data split proportionally into train/val/test.
    """
    if opt.dataset == 'mnist':
        tfm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        full_train = torchvision.datasets.MNIST(
            './data/mnist', train=True,  download=True, transform=tfm)
        test_set   = torchvision.datasets.MNIST(
            './data/mnist', train=False, download=True, transform=tfm)

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

def sample_hyperparameters(trial: optuna.Trial, dataset_name: str, variant: str) -> dict:
    """
    All tuneable parameters in one place.

    Variant-conditional parameters
    --------------------------------
    leaf_dist   : 'jointly_training' is included (controls EM vs gradient leaf update)
    leaf_proto  : 'proto_dim' is included; 'jointly_training' is excluded
    leaf_expert : neither 'jointly_training' nor 'proto_dim' is needed

    Dataset-conditional parameters
    --------------------------------
    mnist : CNN architecture knobs (n_conv_blocks, base_channels, kernel_size)
    UCI   : MLP architecture knobs (n_layers, hidden_size)

    Memory warning
    --------------
    n_leaf = 2^tree_depth.
    """
    hp = {
        # --- Forest ---
        'n_tree':            trial.suggest_int  ('n_tree',            2,   80),
        'tree_depth':        trial.suggest_int  ('tree_depth',        2,   10),
        'tree_feature_rate': trial.suggest_float('tree_feature_rate', 0.1,  1.0),
        # --- Optimiser ---
        'lr':           trial.suggest_float('lr',           1e-4, 1e-1, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
    }

    # leaf_dist only: whether to train π by gradient or EM
    if variant == 'leaf_dist':
        hp['jointly_training'] = trial.suggest_categorical('jointly_training', [True, False])
    else:
        hp['jointly_training'] = True   # always gradient for proto/expert

    # leaf_proto only: prototype vector dimensionality
    if variant == 'leaf_proto':
        hp['proto_dim'] = trial.suggest_categorical('proto_dim', [64, 128, 256, 512])
    else:
        hp['proto_dim'] = 128           # placeholder, not used

    # --- Feature extractor (dataset-conditional) ---
    if dataset_name == 'mnist':
        hp['n_conv_blocks'] = trial.suggest_int('n_conv_blocks', 1, 4)
        hp['base_channels'] = trial.suggest_categorical('base_channels', [16, 32, 64])
        hp['kernel_size']   = trial.suggest_categorical('kernel_size',   [3, 5])
        hp['dropout_rate']  = trial.suggest_float('dropout_rate', 0.0, 0.5)
        hp['batch_norm']    = trial.suggest_categorical('batch_norm',    [True, False])
    else:
        hp['n_layers']     = trial.suggest_int('n_layers',    1,  3)
        hp['hidden_size']  = trial.suggest_categorical('hidden_size', [256, 512, 1024, 2048])
        hp['dropout_rate'] = trial.suggest_float('dropout_rate', 0.0, 0.5)
        hp['batch_norm']   = trial.suggest_categorical('batch_norm',    [True, False])

    return hp


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(hp: dict, dataset_name: str, n_class: int, variant: str,
                device: torch.device) -> torch.nn.Module:
    """
    Build the requested NDF variant from a hyperparameter dict.
    Works for both Optuna trial dicts and final best_params dicts.
    """

    # --- feature extractor ---
    if dataset_name == 'mnist':
        feat_layer = ndf.MNISTFeatureLayer(
            dropout_rate  = hp['dropout_rate'],
            n_conv_blocks = hp['n_conv_blocks'],
            base_channels = hp['base_channels'],
            kernel_size   = hp['kernel_size'],
            batch_norm    = hp['batch_norm'],
        )
    else:
        feat_cls = {
            'adult':  ndf.UCIAdultFeatureLayer,
            'letter': ndf.UCILetterFeatureLayer,
            'yeast':  ndf.UCIYeastFeatureLayer,
        }[dataset_name]
        feat_layer = feat_cls(
            dropout_rate = hp['dropout_rate'],
            n_layers     = hp['n_layers'],
            hidden_size  = hp['hidden_size'],
            batch_norm   = hp['batch_norm'],
        )

    n_feat = feat_layer.get_out_feature_size()

    # --- forest + model ---
    if variant == 'leaf_dist':
        forest = ndf.Forest(
            n_tree=hp['n_tree'], tree_depth=hp['tree_depth'],
            n_in_feature=n_feat, tree_feature_rate=hp['tree_feature_rate'],
            n_class=n_class, jointly_training=hp['jointly_training'],
        )
        model = ndf.NeuralDecisionForest(feat_layer, forest)

    elif variant == 'leaf_proto':
        forest = ndf.ProtoForest(
            n_tree=hp['n_tree'], tree_depth=hp['tree_depth'],
            n_in_feature=n_feat, tree_feature_rate=hp['tree_feature_rate'],
            proto_dim=hp['proto_dim'],
        )
        model = ndf.NeuralDecisionForestProto(feat_layer, forest, n_class=n_class)

    elif variant == 'leaf_expert':
        forest = ndf.ExpertForest(
            n_tree=hp['n_tree'], tree_depth=hp['tree_depth'],
            n_in_feature=n_feat, tree_feature_rate=hp['tree_feature_rate'],
            n_class=n_class,
        )
        model = ndf.NeuralDecisionForestExpert(feat_layer, forest)

    else:
        raise NotImplementedError(f"Unknown variant: {variant}")

    return model.to(device)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _pi_em_update(model, loader, device):
    """
    Convex EM update of leaf distributions for fixed routing (Eq. 17).
    Only called when variant='leaf_dist' and jointly_training=False.
    """
    n_class    = model.forest.trees[0].n_class
    cls_onehot = torch.eye(n_class, device=device)
    feat_batches, target_batches = [], []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            feats = model.feature_layer(data).view(data.size(0), -1)
            feat_batches.append(feats)
            target_batches.append(cls_onehot[target])

        for tree in tqdm(model.forest.trees, desc='EM update π'):
            mu_batches = [tree(feats) for feats in feat_batches]

            for _ in range(20):
                new_pi = torch.zeros(tree.n_leaf, tree.n_class, device=device)
                for mu, target_oh in zip(mu_batches, target_batches):
                    pi   = tree.get_pi()
                    prob = tree.cal_prob(mu, pi)

                    _target = target_oh.unsqueeze(1)
                    _pi     = pi.unsqueeze(0)
                    _mu     = mu.unsqueeze(2)
                    _prob   = torch.clamp(prob.unsqueeze(1), min=1e-6, max=1.)

                    new_pi += (torch.mul(torch.mul(_target, _pi), _mu) / _prob).sum(dim=0)

                tree.update_pi(F.softmax(new_pi, dim=1))


def train_one_epoch(model, loader, optim, device, variant, jointly_training) -> None:
    """
    Run one SGD pass.

    For leaf_dist + jointly_training=False the EM π update is run first
    (same loader object — we iterate it twice, which is fine for small datasets).
    For leaf_proto and leaf_expert the EM block is skipped entirely.
    """
    # EM leaf update (leaf_dist two-stage only)
    if variant == 'leaf_dist' and not jointly_training:
        _pi_em_update(model, loader, device)

    # SGD pass — identical for all three variants
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optim.zero_grad()
        output = model(data)
        F.nll_loss(torch.log(output + 1e-8), target).backward()
        optim.step()


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """
    Return both metrics in a single forward pass.

    Returns
    -------
    {'val_loss': float, 'val_acc': float}
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output      = model(data)
        total_loss += F.nll_loss(torch.log(output + 1e-8), target, reduction='sum').item()
        correct    += output.argmax(dim=1).eq(target).sum().item()
        total      += target.size(0)
    return {'val_loss': total_loss / total, 'val_acc': correct / total}


# ---------------------------------------------------------------------------
# Optuna trial
# ---------------------------------------------------------------------------

def run_trial(trial: optuna.Trial, db: dict, opt, device: torch.device) -> float:
    """
    Train one hyperparameter configuration for up to opt.epochs epochs.

    Two complementary stopping mechanisms run in parallel:
    1. EarlyStopping     — saves best checkpoint; halts on plateau
    2. Optuna pruner     — kills clearly underperforming trials early
                           (always uses val_acc regardless of es_monitor)

    Returns best val_acc at the restored best-checkpoint epoch.
    """
    hp    = sample_hyperparameters(trial, opt.dataset, opt.variant)
    model = build_model(hp, opt.dataset, opt.n_class, opt.variant, device)
    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=hp['lr'], weight_decay=hp['weight_decay'],
    )

    train_loader = DataLoader(db['train'], batch_size=256, shuffle=True)
    val_loader   = DataLoader(db['val'],   batch_size=256, shuffle=False)

    es_mode = 'min' if opt.es_monitor == 'val_loss' else 'max'
    es = EarlyStopping(mode=es_mode, patience=opt.es_patience, min_delta=opt.es_min_delta)

    try:
        for epoch in range(1, opt.epochs + 1):
            train_one_epoch(model, train_loader, optim, device,
                            opt.variant, hp['jointly_training'])
            metrics = evaluate(model, val_loader, device)

            es.step(metrics[opt.es_monitor], model)

            trial.report(metrics['val_acc'], epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if es.should_stop:
                logging.info(
                    f"Trial {trial.number}: early stopping at epoch {epoch} "
                    f"(best {opt.es_monitor} = {es.best_value:.4f})"
                )
                break

        es.restore_best(model)
        return evaluate(model, val_loader, device)['val_acc']

    finally:
        es.cleanup()


# ---------------------------------------------------------------------------
# Final training
# ---------------------------------------------------------------------------

def final_training(best_params: dict, db: dict, opt, device: torch.device) -> float:
    """
    Retrain from scratch with the best hyperparameters on train ∪ val.

    Training : train ∪ val
    ES        : val split (checkpoint selection only)
    Test      : touched here for the first and only time

    Returns final test accuracy.
    """
    print("\n" + "=" * 60)
    print(f"Final retraining  |  variant={opt.variant}  "
          f"es_monitor={opt.es_monitor}")
    print("=" * 60)

    if opt.variant != 'leaf_dist':
        print("\nNote: leaf_proto and leaf_expert variants always use jointly_training=True.")
        best_params['jointly_training'] = True

    model = build_model(best_params, opt.dataset, opt.n_class, opt.variant, device)
    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=best_params['lr'], weight_decay=best_params['weight_decay'],
    )

    train_loader = DataLoader(
        ConcatDataset([db['train'], db['val']]), batch_size=256, shuffle=True)
    val_loader   = DataLoader(db['val'],  batch_size=256, shuffle=False)
    test_loader  = DataLoader(db['test'], batch_size=256, shuffle=False)

    es_mode = 'min' if opt.es_monitor == 'val_loss' else 'max'
    es = EarlyStopping(mode=es_mode, patience=opt.es_patience, min_delta=opt.es_min_delta)

    try:
        for epoch in range(1, opt.epochs + 1):
            train_one_epoch(model, train_loader, optim, device,
                            opt.variant, best_params['jointly_training'])
            metrics = evaluate(model, val_loader, device)
            es.step(metrics[opt.es_monitor], model)

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
        val_m  = evaluate(model, val_loader,  device)
        test_m = evaluate(model, test_loader, device)

        print(f"\n  Best checkpoint — val_loss: {val_m['val_loss']:.4f}"
              f"  |  val_acc: {val_m['val_acc']:.4f}")
        print(f"  Test set        — test_loss: {test_m['val_loss']:.4f}"
              f"  |  test_acc: {test_m['val_acc']:.4f}")
        print("=" * 60)

        # Save with variant metadata for transfer.py
        torch.save({
            'model_state_dict': model.state_dict(),
            'hyperparameters':  best_params,
            'variant':          opt.variant,
            'n_class':          opt.n_class,
            'dataset':          opt.dataset,
        }, opt.save_path)
        print(f"\nModel saved to: {opt.save_path}")

        return test_m['val_acc']

    finally:
        es.cleanup()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s')
    opt = parse_arg()

    device = (
        torch.device('cuda', opt.gpuid)
        if opt.gpuid >= 0 and torch.cuda.is_available()
        else torch.device('cpu')
    )
    print(f"Device      : {device}")
    print(f"Variant     : {opt.variant}")
    print(f"ES monitor  : {opt.es_monitor}  "
          f"(patience={opt.es_patience}, min_delta={opt.es_min_delta})")

    db = prepare_db(opt)
    print(f"Split sizes — train: {len(db['train'])}  "
          f"val: {len(db['val'])}  test: {len(db['test'])}")

    study = optuna.create_study(
        study_name=opt.study_name,
        direction='maximize',
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

    print("\n" + "=" * 60)
    print("Hyperparameter search complete.")
    print(f"  Best val accuracy (tuning) : {study.best_value:.4f}")
    print("  Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k:<22} = {v}")

    csv_path = f"{opt.study_name}_trials.csv"
    study.trials_dataframe().to_csv(csv_path, index=False)
    print(f"\nFull trial log saved to: {csv_path}")

    test_acc = final_training(study.best_params, db, opt, device)
    print(f"\nFinal held-out test accuracy: {test_acc:.4f}")

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
