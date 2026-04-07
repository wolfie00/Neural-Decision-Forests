import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
from collections import OrderedDict
from typing import cast
import torch.nn.functional as F
 
 
class MNISTFeatureLayer(nn.Sequential):
    def __init__(self, dropout_rate, shallow=False):
        super(MNISTFeatureLayer, self).__init__()
        self.shallow = shallow
        if shallow:
            self.add_module('conv1', nn.Conv2d(1, 64, kernel_size=15, padding=1, stride=5))
        else:
            self.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, padding=1))
            self.add_module('relu1', nn.ReLU())
            self.add_module('pool1', nn.MaxPool2d(kernel_size=2))
            self.add_module('drop1', nn.Dropout(dropout_rate))
            self.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1))
            self.add_module('relu2', nn.ReLU())
            self.add_module('pool2', nn.MaxPool2d(kernel_size=2))
            self.add_module('drop2', nn.Dropout(dropout_rate))
            self.add_module('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1))
            self.add_module('relu3', nn.ReLU())
            self.add_module('pool3', nn.MaxPool2d(kernel_size=2))
            self.add_module('drop3', nn.Dropout(dropout_rate))
 
    def get_out_feature_size(self):
        if self.shallow:
            return 64 * 4 * 4
        else:
            return 128 * 3 * 3
 
 
class UCIAdultFeatureLayer(nn.Sequential):
    def __init__(self, dropout_rate=0., shallow=True):
        super(UCIAdultFeatureLayer, self).__init__()
        self.shallow = shallow
        if shallow:
            self.add_module('linear', nn.Linear(113, 1024))
        else:
            raise NotImplementedError
 
    def get_out_feature_size(self):
        return 1024
 
 
class UCILetterFeatureLayer(nn.Sequential):
    def __init__(self, dropout_rate=0., shallow=True):
        super(UCILetterFeatureLayer, self).__init__()
        self.shallow = shallow
        if shallow:
            self.add_module('linear', nn.Linear(16, 1024))
        else:
            raise NotImplementedError
 
    def get_out_feature_size(self):
        return 1024
 
 
class UCIYeastFeatureLayer(nn.Sequential):
    def __init__(self, dropout_rate=0., shallow=True):
        super(UCIYeastFeatureLayer, self).__init__()
        self.shallow = shallow
        if shallow:
            self.add_module('linear', nn.Linear(8, 1024))
        else:
            raise NotImplementedError
 
    def get_out_feature_size(self):
        return 1024
 
 
class Tree(nn.Module):
    def __init__(self, depth, n_in_feature, used_feature_rate, n_class, jointly_training=True):
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.n_class = n_class
        self.jointly_training = jointly_training
 
        # Used features in this tree
        n_used_feature = int(n_in_feature * used_feature_rate)
        onehot = np.eye(n_in_feature)
        using_idx = np.random.choice(np.arange(n_in_feature), n_used_feature, replace=False)
        feature_mask = onehot[using_idx].T  # shape: [n_in_feature, n_used_feature]
        feature_mask = torch.from_numpy(feature_mask).float()
 
        # CHANGED: use register_buffer instead of Parameter(requires_grad=False)
        # Buffers are part of model state, move with .to(device), but are not optimized.
        self.register_buffer('feature_mask', feature_mask)
        self.feature_mask: torch.Tensor  # type annotation for Pylance
 
        # Leaf label distribution
        if jointly_training:
            pi = np.random.rand(self.n_leaf, n_class)
            self.pi = Parameter(torch.from_numpy(pi).float(), requires_grad=True)
        else:
            pi = np.ones((self.n_leaf, n_class)) / n_class
            # CHANGED: non-trainable pi also becomes a buffer for consistency
            self.register_buffer('pi', torch.from_numpy(pi).float())
 
        # Decision network
        self.decision = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(n_used_feature, self.n_leaf)),
            ('sigmoid', nn.Sigmoid()),
        ]))
 
    def forward(self, x):
        """
        :param x (Tensor): [batch_size, n_features]
        :return: route probability (Tensor): [batch_size, n_leaf]
        """
        # CHANGED: feature_mask is now a buffer and automatically on the correct device;
        # no manual .cuda() check is needed.
        feats = torch.mm(x, self.feature_mask)  # -> [batch_size, n_used_feature]
        decision = self.decision(feats)          # -> [batch_size, n_leaf]
 
        decision = torch.unsqueeze(decision, dim=2)
        decision_comp = 1 - decision
        decision = torch.cat((decision, decision_comp), dim=2)  # -> [batch_size, n_leaf, 2]
 
        # Compute route probability
        batch_size = x.size(0)
 
        # CHANGED: replaced Variable(x.data.new(...).fill_(1.)) with torch.ones(...)
        # Variable is fully deprecated; plain tensors support autograd since PyTorch 0.4.
        _mu = torch.ones(batch_size, 1, 1, dtype=x.dtype, device=x.device)
 
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _decision = decision[:, begin_idx:end_idx, :]  # -> [batch_size, 2**n_layer, 2]
            _mu = _mu * _decision                          # -> [batch_size, 2**n_layer, 2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer + 1)
 
        mu = _mu.view(batch_size, self.n_leaf)
        return mu
 
    def get_pi(self):
        if self.jointly_training:
            return F.softmax(self.pi, dim=-1)
        else:
            return self.pi
 
    def cal_prob(self, mu, pi):
        """
        :param mu:  [batch_size, n_leaf]
        :param pi:  [n_leaf, n_class]
        :return:    label probability [batch_size, n_class]
        """
        p = torch.mm(mu, pi)
        return p
 
    def update_pi(self, new_pi):
        self.pi.data = new_pi
 
 
class Forest(nn.Module):
    def __init__(self, n_tree, tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree = n_tree
        for _ in range(n_tree):
            tree = Tree(tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training)
            self.trees.append(tree)
 
    def forward(self, x):
        probs = []
        for tree in self.trees:
            tree = cast(Tree, tree)  # type annotation for Pylance
            mu = tree(x)
            p = tree.cal_prob(mu, tree.get_pi())
            probs.append(p.unsqueeze(2))
        probs = torch.cat(probs, dim=2)
        prob = torch.sum(probs, dim=2) / self.n_tree
        return prob
 
 
class NeuralDecisionForest(nn.Module):
    def __init__(self, feature_layer, forest):
        super(NeuralDecisionForest, self).__init__()
        self.feature_layer = feature_layer
        self.forest = forest
 
    def forward(self, x):
        out = self.feature_layer(x)
        out = out.view(x.size(0), -1)
        out = self.forest(out)
        return out
 