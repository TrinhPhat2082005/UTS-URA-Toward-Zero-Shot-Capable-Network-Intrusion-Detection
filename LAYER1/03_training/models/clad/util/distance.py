"""
Distance / similarity helpers — copy nguyen tu UTS_Layer1/clad/util/distance.py,
KHONG doi logic. (Goc: ported tu CLOSR, jackwilkie/CLOSR, util/distance.py.)

Paper (Eq. 4) va reference code dinh nghia *rescaled* cosine distance:

    d(z, z') = (1 - z.z') / 2   in [0, 1]

Bounded range nay lam margin m = 1.0 nghia la "antipodal" (Sec. V-B). Dung
(1 - cos) in [0, 2] khong-rescale se pha vo loss geometry, nen giu nguyen
cong thuc reference o day.
"""

import torch
import torch.nn.functional as F
import numpy as np


def cossim(a: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    """Pairwise rescaled cosine distance matrix d = (1 - cos)/2 in [0, 1].

    a : (N, D)
    b : (M, D) or None (defaults to a)
    returns (N, M) distance matrix.
    """
    if b is None:
        b = a
    a_norm = F.normalize(a, p=2.0, dim=-1)
    b_norm = F.normalize(b, p=2.0, dim=-1)
    sim = torch.mm(a_norm, b_norm.T)        # cosine similarity in [-1, 1]
    return (1.0 - sim) / 2.0                 # rescaled distance in [0, 1]


# cosdist is an alias in the reference repo
cosdist = cossim


def get_centroid(z: torch.Tensor) -> torch.Tensor:
    """Mean of L2-normalised embeddings, re-normalised to the unit sphere (Eq. 8)."""
    c = z.mean(dim=0)
    return F.normalize(c, dim=0)


@torch.no_grad()
def get_features(model, loader, device) -> tuple:
    """Extract all embeddings and labels from a DataLoader."""
    model.eval()
    zs, ys = [], []
    for X, y in loader:
        X = X.to(device)
        zs.append(model(X).cpu())
        ys.append(y)
    return torch.cat(zs), torch.cat(ys)


@torch.no_grad()
def cosine_scores_to_centroid(z: torch.Tensor, centroid: torch.Tensor) -> np.ndarray:
    """
    OOD score s(x) = -(z . mu)   (Eq. 9), high = anomaly.

    NB: tra ve *signed cosine similarity* (z.mu) o day; caller tu negate de
    co anomaly score. Similarity thap -> xa benign centroid -> attack.
    """
    return (z @ centroid).numpy()
