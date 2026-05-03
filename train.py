"""
train.py — Simple training script for all three NDF leaf variants.

Variants
--------
  leaf_dist   — original class-distribution leaves (jointly or two-stage)
  leaf_proto  — latent prototype leaves + shared task head
  leaf_expert — local linear expert leaves

Usage examples
--------------
  # Original variant, MNIST, jointly trained
  python train.py -dataset mnist -variant leaf_dist -jointly_training -epochs 20

  # Prototype variant, MNIST
  python train.py -dataset mnist -variant leaf_proto -proto_dim 128 -epochs 20

  # Expert variant, MNIST
  python train.py -dataset mnist -variant leaf_expert -epochs 20

The saved checkpoint includes variant metadata so transfer.py can load it:
  torch.load('ndf_trained.pt') → {'model_state_dict', 'variant', 'n_class',
                                   'dataset', 'hp'}
"""

import argparse
import logging

import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

import dataset
import ndf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arg():
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s]: %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description='train.py — NDF training')
    parser.add_argument('-dataset',  choices=['mnist', 'adult', 'letter', 'yeast'], default='mnist')
    parser.add_argument('-batch_size', type=int, default=128)

    # --- variant ---
    parser.add_argument('-variant', choices=['leaf_dist', 'leaf_proto', 'leaf_expert'],
                        default='leaf_dist',
                        help='Leaf parameterisation: leaf_dist (original) | '
                             'leaf_proto (prototype vectors) | leaf_expert (local linear models)')
    parser.add_argument('-proto_dim', type=int, default=128,
                        help='Prototype vector size (leaf_proto variant only)')

    # --- feature extractor ---
    parser.add_argument('-feat_dropout', type=float, default=0.3)

    # --- forest ---
    parser.add_argument('-n_tree',            type=int,   default=5)
    parser.add_argument('-tree_depth',        type=int,   default=3)
    parser.add_argument('-n_class',           type=int,   default=10)
    parser.add_argument('-tree_feature_rate', type=float, default=0.5)

    # --- training ---
    parser.add_argument('-lr',       type=float, default=0.001)
    parser.add_argument('-gpuid',    type=int,   default=-1)
    parser.add_argument('-epochs',   type=int,   default=10)
    parser.add_argument('-report_every', type=int, default=10)

    # --- leaf_dist only ---
    parser.add_argument('-jointly_training', action='store_true', default=False,
                        help='leaf_dist only: train π jointly via gradient instead of EM')

    # --- output ---
    parser.add_argument('-save_path', type=str, default='ndf_just_trained.pt',
                        help='Where to save the trained model checkpoint')

    opt = parser.parse_args()

    if opt.variant != 'leaf_dist' and not opt.jointly_training:
        # jointly_training flag is irrelevant for other variants; set True silently
        opt.jointly_training = True

    return opt


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_db(opt):
    print("Dataset: %s" % opt.dataset)

    if opt.dataset == 'mnist':
        tfm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = torchvision.datasets.MNIST(
            './data/mnist', train=True,  download=True, transform=tfm)
        eval_dataset  = torchvision.datasets.MNIST(
            './data/mnist', train=False, download=True, transform=tfm)
        return {'train': train_dataset, 'eval': eval_dataset}

    elif opt.dataset == 'adult':
        return {'train': dataset.UCIAdult('./data/uci_adult',   train=True),
                'eval':  dataset.UCIAdult('./data/uci_adult',   train=False)}
    elif opt.dataset == 'letter':
        return {'train': dataset.UCILetter('./data/uci_letter', train=True),
                'eval':  dataset.UCILetter('./data/uci_letter', train=False)}
    elif opt.dataset == 'yeast':
        return {'train': dataset.UCIYeast('./data/uci_yeast',   train=True),
                'eval':  dataset.UCIYeast('./data/uci_yeast',   train=False)}
    else:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def prepare_model(opt):
    """Build and return the requested NDF variant."""

    # --- feature extractor ---
    feat_kwargs = dict(dropout_rate=opt.feat_dropout)
    feat_cls = {
        'mnist':  ndf.MNISTFeatureLayer,
        'adult':  ndf.UCIAdultFeatureLayer,
        'letter': ndf.UCILetterFeatureLayer,
        'yeast':  ndf.UCIYeastFeatureLayer,
    }[opt.dataset]
    feat_layer = feat_cls(**feat_kwargs)
    n_feat     = feat_layer.get_out_feature_size()

    # --- forest + top-level model ---
    if opt.variant == 'leaf_dist':
        forest = ndf.Forest(
            n_tree=opt.n_tree, tree_depth=opt.tree_depth,
            n_in_feature=n_feat, tree_feature_rate=opt.tree_feature_rate,
            n_class=opt.n_class, jointly_training=opt.jointly_training,
        )
        model = ndf.NeuralDecisionForest(feat_layer, forest)

    elif opt.variant == 'leaf_proto':
        forest = ndf.ProtoForest(
            n_tree=opt.n_tree, tree_depth=opt.tree_depth,
            n_in_feature=n_feat, tree_feature_rate=opt.tree_feature_rate,
            proto_dim=opt.proto_dim,
        )
        model = ndf.NeuralDecisionForestProto(feat_layer, forest, n_class=opt.n_class)

    elif opt.variant == 'leaf_expert':
        forest = ndf.ExpertForest(
            n_tree=opt.n_tree, tree_depth=opt.tree_depth,
            n_in_feature=n_feat, tree_feature_rate=opt.tree_feature_rate,
            n_class=opt.n_class,
        )
        model = ndf.NeuralDecisionForestExpert(feat_layer, forest)

    else:
        raise NotImplementedError(f"Unknown variant: {opt.variant}")

    model = model.to(opt.device)
    print(model)
    print(f"Variant            : {opt.variant}")
    print(f"Trainable params   : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def prepare_optim(model, opt):
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=opt.lr, weight_decay=1e-5,
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _pi_em_update(model, loader, opt):
    """
    Convex EM update of leaf distributions for fixed routing (Eq. 17).
    Only called when variant='leaf_dist' and jointly_training=False.
    """
    cls_onehot   = torch.eye(opt.n_class, device=opt.device)
    feat_batches, target_batches = [], []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(opt.device), target.to(opt.device)
            feats = model.feature_layer(data).view(data.size(0), -1)
            feat_batches.append(feats)
            target_batches.append(cls_onehot[target])

        for tree in tqdm(model.forest.trees, desc='EM update π'):
            mu_batches = [tree(feats) for feats in feat_batches]

            for _ in range(20):
                new_pi = torch.zeros(tree.n_leaf, tree.n_class, device=opt.device)
                for mu, target_oh in zip(mu_batches, target_batches):
                    pi   = tree.get_pi()
                    prob = tree.cal_prob(mu, pi)

                    _target = target_oh.unsqueeze(1)
                    _pi     = pi.unsqueeze(0)
                    _mu     = mu.unsqueeze(2)
                    _prob   = torch.clamp(prob.unsqueeze(1), min=1e-6, max=1.)

                    new_pi += (torch.mul(torch.mul(_target, _pi), _mu) / _prob).sum(dim=0)

                tree.update_pi(F.softmax(new_pi, dim=1))


def train_one_epoch(model, loader, optim, opt, batch_offset=0):
    """
    Run one SGD epoch.  For leaf_dist + jointly_training=False, the EM leaf
    update is run first (call _pi_em_update before this function) because
    the loader is shared.

    Returns the final batch loss for logging.
    """
    model.train()
    last_loss = 0.0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(opt.device), target.to(opt.device)
        optim.zero_grad()
        output = model(data)
        loss   = F.nll_loss(torch.log(output + 1e-8), target)
        loss.backward()
        optim.step()
        last_loss = loss.item()
        if batch_idx % opt.report_every == 0:
            total = len(loader.dataset)
            seen  = (batch_idx + batch_offset) * len(data)
            print(f'  Batch [{seen}/{total}]  loss = {last_loss:.6f}')
    return last_loss


@torch.no_grad()
def evaluate(model, loader, device):
    """Return {'loss': float, 'acc': float} on the given loader."""
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
    return {'loss': total_loss / total, 'acc': correct / total}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(model, optim, db, opt):
    train_loader = torch.utils.data.DataLoader(
        db['train'], batch_size=opt.batch_size, shuffle=True)
    eval_loader  = torch.utils.data.DataLoader(
        db['eval'],  batch_size=opt.batch_size, shuffle=False)

    for epoch in range(1, opt.epochs + 1):
        print(f'\nEpoch {epoch}/{opt.epochs}  [{opt.variant}]')

        # leaf_dist two-stage: EM leaf update before SGD
        if opt.variant == 'leaf_dist' and not opt.jointly_training:
            print("  Two-stage: updating π (EM)…")
            _pi_em_update(model, train_loader, opt)

        train_one_epoch(model, train_loader, optim, opt)

        metrics = evaluate(model, eval_loader, opt.device)
        print(f'  Eval — loss: {metrics["loss"]:.4f}  acc: {metrics["acc"]:.4f}')

    return model


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_model(model, opt):
    """
    Save the model state dict alongside variant metadata.
    transfer.py expects this exact format to build a compatible target model.
    """
    hp = {
        'n_tree':            opt.n_tree,
        'tree_depth':        opt.tree_depth,
        'tree_feature_rate': opt.tree_feature_rate,
        'feat_dropout':      opt.feat_dropout,
        'jointly_training':  opt.jointly_training,
        'proto_dim':         opt.proto_dim,
    }
    payload = {
        'model_state_dict': model.state_dict(),
        'variant':  opt.variant,
        'n_class':  opt.n_class,
        'dataset':  opt.dataset,
        'hp':       hp,
    }
    torch.save(payload, opt.save_path)
    print(f'\nModel saved to: {opt.save_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    opt = parse_arg()

    if opt.gpuid >= 0 and torch.cuda.is_available():
        opt.device = torch.device('cuda', opt.gpuid)
    else:
        opt.device = torch.device('cpu')
        print("WARNING: running on CPU")

    db    = prepare_db(opt)
    model = prepare_model(opt)
    optim = prepare_optim(model, opt)
    train(model, optim, db, opt)
    save_model(model, opt)


if __name__ == '__main__':
    main()