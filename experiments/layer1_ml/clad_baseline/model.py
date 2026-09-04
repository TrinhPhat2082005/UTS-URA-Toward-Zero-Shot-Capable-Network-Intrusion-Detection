"""
ContrastiveMLP — ported from CLOSR (jackwilkie/CLOSR, model/model.py) and
matching Sec. III-E of Wilkie et al., IEEE TNSM 2026.

Architecture (paper III-E):
  - Linear up-projection to width d_model
  - n_layers × [Linear(d_model → d_model) → ReLU]   (fixed width, ReLU)
  - Linear projection head → embedding dim d_out
  - L2-normalise output onto the unit hypersphere

The reference uses NO normalisation layer inside the dense blocks by default
(norm_layer defaults to Identity); only the final output is L2-normalised.
We follow that here (block_norm=False) to match the paper, since LayerNorm
inside every block was an addition that is not part of CLAD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    """Linear → activation → (optional norm) → (optional dropout).

    Matches the reference DenseBlock ordering: Linear, activation, norm, dropout.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0,
                 block_norm: bool = False, bias: bool = True):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim, bias=bias), nn.ReLU(inplace=True)]
        if block_norm:
            layers.append(nn.LayerNorm(out_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ContrastiveMLP(nn.Module):
    """
    MLP encoder for CLAD anomaly detection.

    Parameters
    ----------
    d_in        : input feature dimension (78 for this CIC-IDS split)
    hidden_dim  : fixed width of all hidden layers (d_model)
    n_layers    : number of DenseBlocks after the up-projection
    d_out       : embedding dimension (projection head output)
    dropout     : dropout probability in hidden blocks
    block_norm  : if True, add LayerNorm inside each block (NOT the paper default)
    """

    def __init__(self, d_in: int = 78, hidden_dim: int = 512,
                 n_layers: int = 4, d_out: int = 32, dropout: float = 0.0,
                 block_norm: bool = False):
        super().__init__()

        # Up-projection to d_model
        blocks = [DenseBlock(d_in, hidden_dim, dropout=dropout, block_norm=block_norm)]
        # Fixed-width body
        for _ in range(n_layers - 1):
            blocks.append(DenseBlock(hidden_dim, hidden_dim, dropout=dropout,
                                     block_norm=block_norm))
        self.encoder = nn.Sequential(*blocks)

        # Projection head onto the embedding dim (no bias — contrastive practice)
        self.projection = nn.Linear(hidden_dim, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1)   # unit-sphere embedding (Sec. III-E)


def build_model(d_in: int = 78, hidden_dim: int = 512, n_layers: int = 4,
                d_out: int = 32, dropout: float = 0.0,
                block_norm: bool = False) -> ContrastiveMLP:
    return ContrastiveMLP(d_in=d_in, hidden_dim=hidden_dim,
                          n_layers=n_layers, d_out=d_out, dropout=dropout,
                          block_norm=block_norm)
