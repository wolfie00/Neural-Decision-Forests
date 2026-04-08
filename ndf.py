import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
from collections import OrderedDict
from typing import cast
import torch.nn.functional as F
 
 
# ---------------------------------------------------------------------------
# Feature layers
# ---------------------------------------------------------------------------
 
class MNISTFeatureLayer(nn.Module):
    """
    Configurable CNN feature extractor for 1-channel 28×28 images (MNIST).
 
    Architecture
    ------------
    n_conv_blocks stacked blocks, each containing:
        Conv2d(in → out, kernel_size, padding) → BatchNorm2d → ReLU
        → MaxPool2d(2) → Dropout2d(dropout_rate)
 
    Channel progression: base_channels, base_channels×2, base_channels×4, …
    The output spatial size shrinks by ×2 per block (MaxPool).
 
    Parameters
    ----------
    dropout_rate   : dropout probability applied after each pooling step.
    n_conv_blocks  : number of conv blocks (1–4 recommended for 28×28 input).
    base_channels  : channels in the first block; doubles each subsequent block.
    kernel_size    : conv kernel size; must be odd (padding = kernel_size // 2
                     is set automatically to preserve spatial dims before pooling).
 
    Notes
    -----
    - Output feature size is determined by a dummy forward pass in __init__
      so get_out_feature_size() is always correct regardless of architecture.
    - The original shallow / deep distinction is replaced by n_conv_blocks=1
      (shallow) or n_conv_blocks≥2 (deep).
    """
 
    def __init__(
        self,
        dropout_rate: float = 0.0,
        n_conv_blocks: int = 3,
        base_channels: int = 32,
        kernel_size: int = 3,
        batch_norm: bool = True,
    ):
        super().__init__()
        if n_conv_blocks < 1:
            raise ValueError("n_conv_blocks must be ≥ 1")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
 
        padding = kernel_size // 2
        blocks  = []
        in_ch   = 1
 
        for i in range(n_conv_blocks):
            out_ch = base_channels * (2 ** i)
            blocks += [
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_ch) if batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout2d(dropout_rate),
            ]
            in_ch = out_ch
 
        self.features = nn.Sequential(*blocks)
 
        # Determine output size once via a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            self._out_size = int(self.features(dummy).view(1, -1).shape[1])
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
 
    def get_out_feature_size(self) -> int:
        return self._out_size
 
 
class _UCIFeatureLayer(nn.Module):
    """
    Shared configurable MLP feature extractor for tabular UCI datasets.
 
    Architecture
    ------------
    n_layers fully-connected blocks, each containing:
        Linear(in → hidden_size) → BatchNorm1d → ReLU → Dropout
 
    Parameters
    ----------
    input_size   : dimensionality of the raw input features.
    dropout_rate : dropout probability applied in every block.
    n_layers     : number of Linear blocks (≥ 1).
    hidden_size  : width of every hidden layer (constant across all blocks).
    """
 
    def __init__(
        self,
        input_size: int,
        dropout_rate: float = 0.,
        n_layers: int = 1,
        hidden_size: int = 1024,
        batch_norm: bool = True,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be ≥ 1")
 
        blocks = []
        in_size = input_size
        for _ in range(n_layers):
            blocks += [
                nn.Linear(in_size, hidden_size),
                nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ]
            in_size = hidden_size
 
        self.features   = nn.Sequential(*blocks)
        self._out_size  = hidden_size
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
 
    def get_out_feature_size(self) -> int:
        return self._out_size
 
 
class UCIAdultFeatureLayer(_UCIFeatureLayer):
    """MLP feature extractor for UCI Adult (113 input features)."""
    def __init__(self, dropout_rate: float = 0., n_layers: int = 1, hidden_size: int = 1024, 
                 batch_norm: bool = True):
        super().__init__(input_size=113, dropout_rate=dropout_rate,
                         n_layers=n_layers, hidden_size=hidden_size, batch_norm=batch_norm)
 
 
class UCILetterFeatureLayer(_UCIFeatureLayer):
    """MLP feature extractor for UCI Letter (16 input features)."""
    def __init__(self, dropout_rate: float = 0., n_layers: int = 1, hidden_size: int = 1024, 
                 batch_norm: bool = True):
        super().__init__(input_size=16, dropout_rate=dropout_rate,
                         n_layers=n_layers, hidden_size=hidden_size, batch_norm=batch_norm)
 
 
class UCIYeastFeatureLayer(_UCIFeatureLayer):
    """MLP feature extractor for UCI Yeast (8 input features)."""
    def __init__(self, dropout_rate: float = 0., n_layers: int = 1, hidden_size: int = 1024, 
                 batch_norm: bool = True):
        super().__init__(input_size=8, dropout_rate=dropout_rate,
                         n_layers=n_layers, hidden_size=hidden_size, batch_norm=batch_norm)
 
 
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
 