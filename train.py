import argparse
import logging
 
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
 
import dataset
import ndf
 
 
def parse_arg():
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s]: %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-dataset', choices=['mnist', 'adult', 'letter', 'yeast'], default='mnist')
    parser.add_argument('-batch_size', type=int, default=128)
 
    parser.add_argument('-feat_dropout', type=float, default=0.3)
 
    parser.add_argument('-n_tree', type=int, default=5)
    parser.add_argument('-tree_depth', type=int, default=3)
    parser.add_argument('-n_class', type=int, default=10)
    parser.add_argument('-tree_feature_rate', type=float, default=0.5)
 
    parser.add_argument('-lr', type=float, default=0.001, help="sgd: 10, adam: 0.001")
    parser.add_argument('-gpuid', type=int, default=-1)
    parser.add_argument('-jointly_training', action='store_true', default=False)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-report_every', type=int, default=10)
 
    opt = parser.parse_args()
    return opt
 
 
def prepare_db(opt):
    print("Use %s dataset" % (opt.dataset))
 
    if opt.dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST('./data/mnist', train=True, download=True,
                                                   transform=torchvision.transforms.Compose([
                                                       torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                   ]))
        eval_dataset = torchvision.datasets.MNIST('./data/mnist', train=False, download=True,
                                                  transform=torchvision.transforms.Compose([
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                  ]))
        return {'train': train_dataset, 'eval': eval_dataset}
 
    elif opt.dataset == 'adult':
        train_dataset = dataset.UCIAdult('./data/uci_adult', train=True)
        eval_dataset = dataset.UCIAdult('./data/uci_adult', train=False)
        return {'train': train_dataset, 'eval': eval_dataset}
 
    elif opt.dataset == 'letter':
        train_dataset = dataset.UCILetter('./data/uci_letter', train=True)
        eval_dataset = dataset.UCILetter('./data/uci_letter', train=False)
        return {'train': train_dataset, 'eval': eval_dataset}
 
    elif opt.dataset == 'yeast':
        train_dataset = dataset.UCIYeast('./data/uci_yeast', train=True)
        eval_dataset = dataset.UCIYeast('./data/uci_yeast', train=False)
        return {'train': train_dataset, 'eval': eval_dataset}
 
    else:
        raise NotImplementedError
 
 
def prepare_model(opt):
    if opt.dataset == 'mnist':
        feat_layer = ndf.MNISTFeatureLayer(opt.feat_dropout)
    elif opt.dataset == 'adult':
        feat_layer = ndf.UCIAdultFeatureLayer(opt.feat_dropout)
    elif opt.dataset == 'letter':
        feat_layer = ndf.UCILetterFeatureLayer(opt.feat_dropout)
    elif opt.dataset == 'yeast':
        feat_layer = ndf.UCIYeastFeatureLayer(opt.feat_dropout)
    else:
        raise NotImplementedError
 
    forest = ndf.Forest(
        n_tree=opt.n_tree,
        tree_depth=opt.tree_depth,
        n_in_feature=feat_layer.get_out_feature_size(),
        tree_feature_rate=opt.tree_feature_rate,
        n_class=opt.n_class,
        jointly_training=opt.jointly_training
    )
    model = ndf.NeuralDecisionForest(feat_layer, forest)
 
    # CHANGED: use .to(device) instead of branching .cuda() / .cpu() calls.
    # opt.device is set in main() based on gpuid.
    model = model.to(opt.device)
 
    print(model)
    return model
 
 
def prepare_optim(model, opt):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=opt.lr, weight_decay=1e-5)
 
 
def train(model, optim, db, opt):
    for epoch in range(1, opt.epochs + 1):
 
        # Update \Pi (two-stage learning)
        if not opt.jointly_training:
            print("Epoch %d : Two Stage Learning - Update PI" % epoch)
 
            cls_onehot = torch.eye(opt.n_class, device=opt.device)
            feat_batches = []
            target_batches = []
            train_loader = torch.utils.data.DataLoader(db['train'], batch_size=opt.batch_size, shuffle=True)
 
            with torch.no_grad():
                for data, target in train_loader:
                    # CHANGED: use .to(device) throughout instead of .cuda() conditionals.
                    data = data.to(opt.device)
                    target = target.to(opt.device)
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
                        new_pi = torch.zeros((tree.n_leaf, tree.n_class), device=opt.device)
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
        train_loader = torch.utils.data.DataLoader(db['train'], batch_size=opt.batch_size, shuffle=True)
        for batch_idx, (data, target) in enumerate(train_loader):
            # CHANGED: .to(device) replaces conditional .cuda() calls.
            data = data.to(opt.device)
            target = target.to(opt.device)
            # CHANGED: removed Variable() wrappers.
            optim.zero_grad()
            output = model(data)
            loss = F.nll_loss(torch.log(output), target)
            loss.backward()
            optim.step()
            if batch_idx % opt.report_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
 
        # Eval
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        test_loader = torch.utils.data.DataLoader(db['eval'], batch_size=opt.batch_size, shuffle=True)
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(opt.device)
                target = target.to(opt.device)
                # CHANGED: removed Variable() wrappers.
                output = model(data)
                # CHANGED: removed deprecated size_average=False; replaced with reduction='sum'.
                test_loss += F.nll_loss(torch.log(output), target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)  # CHANGED: .argmax() replaces .max()[1] idiom.
                correct += pred.eq(target.view_as(pred)).sum().item()  # CHANGED: .item() gives a plain int.
                total += target.size(0)
 
        test_loss /= total
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
            test_loss, correct, total,
            correct / total))
 
 
def main():
    opt = parse_arg()
 
    # CHANGED: resolve a single torch.device object and store it on opt.
    # This replaces the scattered opt.cuda boolean + conditional .cuda() calls.
    if opt.gpuid >= 0 and torch.cuda.is_available():
        opt.device = torch.device('cuda', opt.gpuid)
    else:
        opt.device = torch.device('cpu')
        print("WARNING: RUN WITHOUT GPU")
 
    db = prepare_db(opt)
    model = prepare_model(opt)
    optim = prepare_optim(model, opt)
    train(model, optim, db, opt)
 
 
if __name__ == '__main__':
    main()
