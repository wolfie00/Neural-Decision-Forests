"""
Neural Decision Forest: original, prototype, and expert variants.

Three leaf parameterisations are implemented, following the taxonomy in the
transfer-learning note:

  NeuralDecisionForest       — leaves store class distributions   (original)
  NeuralDecisionForestProto  — leaves store latent prototype vectors
  NeuralDecisionForestExpert — leaves store local linear expert models

All three share the same feature extractors and the same differentiable
routing logic, which is factored out into _BaseTree to avoid repetition.
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F


# ============================================================
# Feature layers
# ============================================================

class MNISTFeatureLayer(nn.Module):
    """
    Configurable CNN feature extractor for 1-channel 28×28 images (MNIST).

    Architecture
    ------------
    n_conv_blocks stacked blocks, each containing:
        Conv2d → BatchNorm2d → ReLU → MaxPool2d(2) → Dropout2d

    Channel progression: base_channels, base_channels×2, base_channels×4, …
    Each MaxPool halves the spatial size; output size is inferred once via a
    dummy forward pass so get_out_feature_size() is always correct.

    Parameters
    ----------
    dropout_rate   : Dropout2d probability applied after each pooling step.
    n_conv_blocks  : Number of conv blocks (1–4 recommended for 28×28 input).
    base_channels  : Channels in the first block; doubled every subsequent block.
    kernel_size    : Conv filter size (must be odd; padding is set automatically).
    """

    def __init__(
        self,
        dropout_rate: float,
        n_conv_blocks: int = 3,
        base_channels: int = 32,
        kernel_size: int = 3,
        batch_norm: bool = True
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
        Linear → BatchNorm1d → ReLU → Dropout

    Parameters
    ----------
    input_size   : Dimensionality of the raw input features.
    dropout_rate : Dropout probability applied in every block.
    n_layers     : Number of Linear blocks (≥ 1).
    hidden_size  : Width of every hidden layer (constant across all blocks).
    """

    def __init__(
        self,
        input_size: int,
        dropout_rate: float = 0.,
        n_layers: int = 1,
        hidden_size: int = 1024,
        batch_norm: bool = True
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be ≥ 1")

        blocks  = []
        in_size = input_size
        for _ in range(n_layers):
            blocks += [
                nn.Linear(in_size, hidden_size),
                nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ]
            in_size = hidden_size

        self.features  = nn.Sequential(*blocks)
        self._out_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def get_out_feature_size(self) -> int:
        return self._out_size


class UCIAdultFeatureLayer(_UCIFeatureLayer):
    """MLP feature extractor for UCI Adult (113 input features)."""
    def __init__(self, dropout_rate: float = 0., n_layers: int = 1, hidden_size: int = 1024):
        super().__init__(113, dropout_rate, n_layers, hidden_size)


class UCILetterFeatureLayer(_UCIFeatureLayer):
    """MLP feature extractor for UCI Letter (16 input features)."""
    def __init__(self, dropout_rate: float = 0., n_layers: int = 1, hidden_size: int = 1024):
        super().__init__(16, dropout_rate, n_layers, hidden_size)


class UCIYeastFeatureLayer(_UCIFeatureLayer):
    """MLP feature extractor for UCI Yeast (8 input features)."""
    def __init__(self, dropout_rate: float = 0., n_layers: int = 1, hidden_size: int = 1024):
        super().__init__(8, dropout_rate, n_layers, hidden_size)


# ============================================================
# Base routing tree (shared by all three variants)
# ============================================================

class _BaseTree(nn.Module):
    """
    Shared stochastic differentiable routing logic.

    At each internal node n the model computes a routing probability
    d_n(x) = σ(f_n(x)) (Eq. 3 in the note).  The probability that sample x
    reaches leaf ℓ is the product of d_n or (1 − d_n) along the path
    (Eq. 5).  Routing is computed in a single vectorised pass over all
    tree layers.

    All tree variants (original, prototype, expert) inherit _BaseTree and
    call self._route(x) to obtain leaf arrival probabilities μ.

    Parameters
    ----------
    depth             : Tree depth d; number of leaves = 2^d.
    n_in_feature      : Dimensionality of the flat feature vector fed to the tree.
    used_feature_rate : Fraction of input features used for routing decisions.
                        A fixed random sub-sampling mask is created at init time
                        and stored as a non-trainable buffer.
    """

    def __init__(self, depth: int, n_in_feature: int, used_feature_rate: float):
        super().__init__()
        self.depth  = depth
        self.n_leaf = 2 ** depth

        n_used   = int(n_in_feature * used_feature_rate)
        onehot   = np.eye(n_in_feature)
        idx      = np.random.choice(n_in_feature, n_used, replace=False)
        self.register_buffer(
            'feature_mask',
            torch.from_numpy(onehot[idx].T).float()  # [n_in, n_used]
        )

        # Each node's routing score is a linear function of the sub-sampled features.
        self.decision = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(n_used, self.n_leaf)),
            ('sigmoid', nn.Sigmoid()),
        ]))

    def _route(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute leaf arrival probabilities μ (Eq. 5).

        Parameters
        ----------
        x : [batch, n_in_feature]

        Returns
        -------
        mu : [batch, n_leaf]
        """
        feats    = torch.mm(x, self.feature_mask)           # [batch, n_used]
        decision = self.decision(feats)                     # [batch, n_leaf]

        decision      = decision.unsqueeze(2)               # [batch, n_leaf, 1]
        decision_comp = 1.0 - decision
        decision      = torch.cat((decision, decision_comp), dim=2)  # [batch, n_leaf, 2]

        B   = x.size(0)
        _mu = torch.ones(B, 1, 1, dtype=x.dtype, device=x.device)

        begin, end = 1, 2
        for layer in range(self.depth):
            _mu   = _mu.view(B, -1, 1).repeat(1, 1, 2)
            _mu   = _mu * decision[:, begin:end, :]
            begin = end
            end   = begin + 2 ** (layer + 1)

        return _mu.view(B, self.n_leaf)                     # [batch, n_leaf]


# ============================================================
# Variant 1: Leaf class distributions (original formulation)
# ============================================================

class Tree(_BaseTree):
    """
    Original NDF tree.  Each leaf ℓ stores a class distribution
    π_ℓ ∈ Δ^{C−1} (Eq. 2).

    Prediction: P_T(y|x) = Σ_ℓ μ_ℓ(x) π_ℓy  (Eq. 7).

    When jointly_training=True  : π is a trainable Parameter; softmax is
                                   applied in get_pi() to ensure it lies on
                                   the simplex.
    When jointly_training=False : π is a Buffer updated via the convex EM
                                   rule (Eq. 17); gradient does not flow
                                   through the leaves.
    """

    def __init__(
        self,
        depth: int,
        n_in_feature: int,
        used_feature_rate: float,
        n_class: int,
        jointly_training: bool = True,
    ):
        super().__init__(depth, n_in_feature, used_feature_rate)
        self.n_class          = n_class
        self.jointly_training = jointly_training

        if jointly_training:
            pi = np.random.rand(self.n_leaf, n_class)
            self.pi = Parameter(torch.from_numpy(pi).float(), requires_grad=True)
        else:
            pi = np.ones((self.n_leaf, n_class)) / n_class
            self.register_buffer('pi', torch.from_numpy(pi).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns leaf arrival probabilities μ [batch, n_leaf]."""
        return self._route(x)

    def get_pi(self) -> torch.Tensor:
        """Return the leaf class distribution (softmax-normalised if trainable)."""
        return F.softmax(self.pi, dim=-1) if self.jointly_training else self.pi

    def cal_prob(self, mu: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """
        Tree prediction P_T(y|x) = Σ_ℓ μ_ℓ π_ℓy  (Eq. 7).

        Parameters  mu [batch, n_leaf], pi [n_leaf, n_class]
        Returns     p  [batch, n_class]
        """
        return torch.mm(mu, pi)

    def update_pi(self, new_pi: torch.Tensor) -> None:
        """Assign new leaf distributions (used by the EM update, Eq. 17)."""
        self.pi.data = new_pi


class Forest(nn.Module):
    """Ensemble of Trees.  Prediction is the tree-averaged class probability (Eq. 8)."""

    def __init__(
        self,
        n_tree: int,
        tree_depth: int,
        n_in_feature: int,
        tree_feature_rate: float,
        n_class: int,
        jointly_training: bool,
    ):
        super().__init__()
        self.n_tree = n_tree
        self.trees  = nn.ModuleList([
            Tree(tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training)
            for _ in range(n_tree)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = []
        for tree in self.trees:
            mu = tree(x)
            probs.append(tree.cal_prob(mu, tree.get_pi()).unsqueeze(2))
        return torch.cat(probs, dim=2).sum(dim=2) / self.n_tree  # [batch, n_class]


class NeuralDecisionForest(nn.Module):
    """
    Original NDF (Peter et al., 2015).

    feature_layer(x) → h → Forest(h) → class probabilities
    """

    def __init__(self, feature_layer: nn.Module, forest: Forest):
        super().__init__()
        self.feature_layer = feature_layer
        self.forest        = forest

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_layer(x).view(x.size(0), -1)
        return self.forest(h)


# ============================================================
# Variant 2: Leaf prototype vectors
# ============================================================

class ProtoTree(_BaseTree):
    """
    NDF tree where each leaf ℓ stores a latent prototype vector
    z_ℓ ∈ R^{proto_dim}  (Section 7 of the note).

    The tree produces a routing-weighted prototype sum:
        z(x) = Σ_ℓ μ_ℓ(x) z_ℓ   ∈ R^{proto_dim}  (Eq. 35)

    This is then passed to a shared task head (in NeuralDecisionForestProto)
    to obtain class logits.  The key transfer advantage is that z_ℓ stores
    label-agnostic latent concepts; only the task head is task-specific.

    Parameters
    ----------
    proto_dim : Dimensionality of each leaf prototype vector.
    """

    def __init__(
        self,
        depth: int,
        n_in_feature: int,
        used_feature_rate: float,
        proto_dim: int,
    ):
        super().__init__(depth, n_in_feature, used_feature_rate)
        self.proto_dim  = proto_dim
        # Small random initialisation keeps the routing-weighted average
        # near zero initially, letting the task head learn from scratch.
        self.prototypes = Parameter(torch.randn(self.n_leaf, proto_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns leaf arrival probabilities μ [batch, n_leaf]."""
        return self._route(x)

    def get_weighted_proto(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Routing-weighted prototype sum z(x) = Σ_ℓ μ_ℓ z_ℓ  (Eq. 35).

        Parameters  mu [batch, n_leaf]
        Returns     z  [batch, proto_dim]
        """
        return torch.mm(mu, self.prototypes)


class ProtoForest(nn.Module):
    """
    Ensemble of ProtoTrees.
    forward() returns the tree-averaged weighted prototype sum z̄(x).
    """

    def __init__(
        self,
        n_tree: int,
        tree_depth: int,
        n_in_feature: int,
        tree_feature_rate: float,
        proto_dim: int,
    ):
        super().__init__()
        self.n_tree    = n_tree
        self.proto_dim = proto_dim
        self.trees     = nn.ModuleList([
            ProtoTree(tree_depth, n_in_feature, tree_feature_rate, proto_dim)
            for _ in range(n_tree)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns z̄ [batch, proto_dim] — prototype sum averaged across trees.
        """
        zs = [tree.get_weighted_proto(tree(x)) for tree in self.trees]
        return torch.stack(zs, dim=0).mean(dim=0)   # [batch, proto_dim]


class NeuralDecisionForestProto(nn.Module):
    """
    NDF with prototype leaves  (Section 7).

    Architecture
    ------------
    feature_layer(x) → h → ProtoForest(h) → z̄ → task_head → logits → softmax

    The task_head is the only component that is target-task specific.
    Encoder, routing, and prototypes can all be reused across tasks,
    making this the most transfer-friendly formulation.

    Parameters
    ----------
    feature_layer : Pre-built feature extractor.
    forest        : ProtoForest instance.
    n_class       : Number of target classes.
    """

    def __init__(
        self,
        feature_layer: nn.Module,
        forest: ProtoForest,
        n_class: int,
    ):
        super().__init__()
        self.feature_layer = feature_layer
        self.forest        = forest
        # Single global target head: W_t ∈ R^{n_class × proto_dim}, b_t ∈ R^{n_class}
        self.task_head     = nn.Linear(forest.proto_dim, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h      = self.feature_layer(x).view(x.size(0), -1)
        z      = self.forest(h)                  # [batch, proto_dim]
        logits = self.task_head(z)               # [batch, n_class]
        return F.softmax(logits, dim=-1)


# ============================================================
# Variant 3: Leaf expert models
# ============================================================

class ExpertTree(_BaseTree):
    """
    NDF tree where each leaf ℓ stores a local linear expert model
    (Section 8 of the note):

        q_ℓ(h) = A_ℓ h + b_ℓ,   A_ℓ ∈ R^{n_class × d},  b_ℓ ∈ R^{n_class}

    The tree's output logits are the routing-weighted expert outputs:
        s(x) = Σ_ℓ μ_ℓ(x) q_ℓ(h)   ∈ R^{n_class}   (Eq. 48)

    This is the most expressive variant but also the most parameter-heavy.
    The expert matrices A_ℓ are stored as a single [n_leaf, n_class, d]
    tensor for efficient batched computation via torch.einsum.

    Parameters
    ----------
    n_class : Number of output classes (determines expert output dimension).
    """

    def __init__(
        self,
        depth: int,
        n_in_feature: int,
        used_feature_rate: float,
        n_class: int,
    ):
        super().__init__(depth, n_in_feature, used_feature_rate)
        self.n_class = n_class

        # A_ℓ ∈ R^{n_class × n_in_feature} for each leaf ℓ
        self.expert_A = Parameter(
            torch.randn(self.n_leaf, n_class, n_in_feature) * 0.01
        )
        self.expert_b = Parameter(torch.zeros(self.n_leaf, n_class))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters  x : [batch, n_in_feature]
        Returns     logits : [batch, n_class]  routing-weighted expert outputs
        """
        mu = self._route(x)                         # [batch, n_leaf]

        # expert_out[b, ℓ, c] = Σ_d x[b,d] · A[ℓ,c,d] + b[ℓ,c]
        expert_out = (
            torch.einsum('bd,lcd->blc', x, self.expert_A)   # [batch, n_leaf, n_class]
            + self.expert_b.unsqueeze(0)                     # [1,    n_leaf, n_class]
        )

        # s(x) = Σ_ℓ μ_ℓ q_ℓ(h)  →  [batch, n_class]
        return torch.einsum('bl,blc->bc', mu, expert_out)


class ExpertForest(nn.Module):
    """
    Ensemble of ExpertTrees.
    forward() returns tree-averaged logits [batch, n_class].
    """

    def __init__(
        self,
        n_tree: int,
        tree_depth: int,
        n_in_feature: int,
        tree_feature_rate: float,
        n_class: int,
    ):
        super().__init__()
        self.n_tree  = n_tree
        self.n_class = n_class
        self.trees   = nn.ModuleList([
            ExpertTree(tree_depth, n_in_feature, tree_feature_rate, n_class)
            for _ in range(n_tree)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_list = [tree(x) for tree in self.trees]
        return torch.stack(logits_list, dim=0).mean(dim=0)  # [batch, n_class]


class NeuralDecisionForestExpert(nn.Module):
    """
    NDF with leaf expert models  (Section 8).

    Architecture
    ------------
    feature_layer(x) → h → ExpertForest(h) → avg_logits → softmax

    Each leaf's expert A_ℓ h + b_ℓ is a local linear predictor in the
    encoder's output space.  The routing determines how these local
    predictors are combined.

    Parameters
    ----------
    feature_layer : Pre-built feature extractor.
    forest        : ExpertForest instance.
    """

    def __init__(self, feature_layer: nn.Module, forest: ExpertForest):
        super().__init__()
        self.feature_layer = feature_layer
        self.forest        = forest

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h      = self.feature_layer(x).view(x.size(0), -1)
        logits = self.forest(h)              # [batch, n_class]
        return F.softmax(logits, dim=-1)


# ============================================================
# Transfer learning utility
# ============================================================

def copy_routing_weights(source_tree: _BaseTree, target_tree: _BaseTree) -> None:
    """
    Copy the feature sub-sampling mask and routing decision weights from
    source_tree to target_tree.

    This is used when building a target model from a pretrained source:
    the routing structure (which latent partition of the input space was
    learned on the source task) can often be reused on the target task,
    while the leaf parameters are re-initialised.

    Both trees must have been created with the same depth, n_in_feature,
    and used_feature_rate so that tensor shapes match.
    """
    target_tree.feature_mask.data.copy_(source_tree.feature_mask.data)
    target_tree.decision.linear1.weight.data.copy_(
        source_tree.decision.linear1.weight.data
    )
    target_tree.decision.linear1.bias.data.copy_(
        source_tree.decision.linear1.bias.data
    )