"""
CLADLoss — copy nguyen tu UTS_Layer1/clad/losses/clad_loss.py, KHONG doi logic.
(Goc: ported 1:1 tu CLOSR, jackwilkie/CLOSR, losses/clad_loss.py, Eq. 7 cua
Wilkie et al., IEEE TNSM 2026.)

Key idea (Sec. III-B): CLAD hoc density ratio P(x|y=0) / P(x|y!=0).
Loss chi tinh tren benign anchor (target_class = 0), nhung CAN known-attack
sample trong batch de tao negative (dissimilar) pair — normalise benign
distribution. Train tren benign-only se collapse CLAD thanh compactness/
Deep-SVDD objective don thuan, khong con la CLAD nua.

    d = (1 - cos)/2                                rescaled cosine distance in [0,1]

    sim pairs   : (y_i = y_j = 0)                  benign-benign
    dissim pairs: (y_i = 0, y_j != 0)              benign anchor - attack

    sim_loss    = mean_{sim}   d^2                       (compactness)
    dissim_loss = mean_{dissim} relu(m - d)^2           (margin / repulsion)

    loss = alpha . sim_loss + (1-alpha) . dissim_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clad.util.distance import cossim


def clad_calc(z: torch.Tensor, y: torch.Tensor,
              target_class: int = 0,
              margin: float = 1.0,
              alpha: float = 0.5,
              squared: bool = True,
              distance_metric=cossim,
              eps: float = 1e-8):
    """
    Compute the CLAD loss over a full (benign + attack) batch.

    Parameters
    ----------
    z            : (N, D) L2-normalised embeddings
    y            : (N,)   integer labels, 0 = benign, !=0 = (any) attack class
    target_class : the anchor (benign) class — loss is evaluated only on these rows
    margin       : hinge margin for negative pairs (paper uses m = 1.0)
    alpha        : weight balancing compactness (alpha) vs. repulsion (1-alpha)
    squared      : square the rescaled distance terms (paper: True)
    distance_metric : callable returning a pairwise distance matrix in [0,1]

    Returns
    -------
    loss     : scalar tensor
    frac_pos : fraction of active margin violations (diagnostic)
    """
    # Full pairwise rescaled-cosine distance matrix d in [0, 1]
    dists = distance_metric(z)                      # (N, N)

    y_row = y.unsqueeze(1)                          # (N, 1)
    y_col = y.unsqueeze(0)                          # (1, N)
    eq_pairs = torch.eq(y_row, y_col)              # same label

    is_target = (y == target_class)               # benign anchors
    anchor_row = is_target.unsqueeze(1)           # (N, 1) anchor is benign

    # Positive pairs: benign anchor AND same class (benign-benign), excluding self
    eye = torch.eye(dists.shape[0], dtype=torch.bool, device=z.device)
    sim_mask = eq_pairs & anchor_row & (~eye)

    # Negative pairs: benign anchor AND different class (benign <-> attack)
    dissim_mask = (~eq_pairs) & anchor_row

    # -- Positive (compactness) term --------------------------------------
    sim_d = dists[sim_mask]
    if sim_d.numel() == 0:
        sim_loss = torch.tensor(0.0, device=z.device)
    else:
        sim_term = sim_d ** 2 if squared else sim_d
        sim_loss = sim_term.sum() / (sim_d.numel() + eps)

    # -- Negative (margin / repulsion) term --------------------------------
    dissim_d = dists[dissim_mask]
    if dissim_d.numel() == 0:
        dissim_loss = torch.tensor(0.0, device=z.device)
        frac_pos = 0.0
    else:
        viol = F.relu(margin - dissim_d)
        viol_term = viol ** 2 if squared else viol
        dissim_loss = viol_term.sum() / (dissim_d.numel() + eps)
        frac_pos = (viol > 0).float().mean().item()

    loss = alpha * sim_loss + (1.0 - alpha) * dissim_loss
    return loss, frac_pos


class CLADLoss(nn.Module):
    def __init__(self, target_class: int = 0, m: float = 1.0,
                 alpha: float = 0.5, squared: bool = True,
                 distance_metric=cossim, eps: float = 1e-8):
        super().__init__()
        self.target_class    = target_class
        self.m               = m
        self.alpha           = alpha
        self.squared         = squared
        self.distance_metric = distance_metric
        self.eps             = eps

    def forward(self, z: torch.Tensor, labels: torch.Tensor):
        return clad_calc(z, labels,
                         target_class=self.target_class,
                         margin=self.m, alpha=self.alpha,
                         squared=self.squared,
                         distance_metric=self.distance_metric,
                         eps=self.eps)
