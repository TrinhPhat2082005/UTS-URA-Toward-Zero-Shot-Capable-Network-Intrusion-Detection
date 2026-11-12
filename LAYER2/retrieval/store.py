"""
Tap tham chieu R luc chay: MOT cho tra ca vector SHAP, gia tri tho va nhan cua moi tien le.

    store = ReferenceStore.load(config.REFERENCE_SET_NPZ)
    look = store.lookup(q)            # q: vector SHAP cua alert (95 chieu voi AE)
    look.label, look.flagged, look.winners     # nhan vote 5-NN, co, chi so cac tien le cung nhan thang
    store.phi[look.winners], store.x[look.winners]

Quy tac (da chot, plan.md muc 3):
  - Mot lan tinh khoang cach L2 (float64) tu q toi moi vector cua R.
  - Nhan l* = vote cua 5 lang gieng gan nhat; hoa phieu -> nhan co flow gan nhat. Khong co nhan ve nhi.
  - winners = cac lang gieng trong 5 mang nhan l* (dung cho bang doi chieu C, evidence/contrast.py).
  - s(q) = trung binh khoang cach toi 10 lang gieng gan nhat (moi nhan); co = s > tau_zd
    (tau_zd doc tu <tap R>.meta.json, hieu chinh boi 02_calibrate_novelty.py).

phi, x, labels, row_ids cung thu tu dong: chi so ma kNN tra ve dung chung cho ca bon mang, nen khong
the lech dong giua SHAP va gia tri tho (xem tests/test_store.py).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import novelty

K_VOTE = 5
K_NOVELTY = 10


@dataclass
class Lookup:
    label: str               # nhan thang vote 5-NN
    votes: dict              # {nhan: so phieu} trong 5 lang gieng (ghi log, khong dua VLM)
    winners: np.ndarray      # chi so (trong R) cac lang gieng mang nhan thang, gan nhat truoc
    neighbours: np.ndarray   # chi so 10 lang gieng gan nhat, gan nhat truoc
    distances: np.ndarray    # khoang cach L2 tuong ung
    novelty: float           # s(q)
    tau_zd: float | None
    flagged: bool


class ReferenceStore:
    def __init__(self, phi, labels, x=None, row_ids=None, feature_names=None, base_feature_names=None,
                 tau_zd=None, source=None):
        self.phi = np.asarray(phi, dtype="float64")
        self.labels = np.asarray(labels)
        self.x = None if x is None else np.asarray(x, dtype="float64")
        self.row_ids = None if row_ids is None else np.asarray(row_ids, dtype="int64")
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.base_feature_names = list(base_feature_names) if base_feature_names is not None else None
        self.tau_zd = tau_zd
        self.source = source
        self._sq = (self.phi ** 2).sum(axis=1)

    @classmethod
    def load(cls, npz_path, exclude_labels=(), tau_zd=None):
        """Doc tap R (retrieval/reference_set.py). x/row_ids la None neu R chua co khoa x__/row__."""
        data = np.load(npz_path, allow_pickle=True)
        labs = sorted(k[len("label__"):] for k in data.files if k.startswith("label__"))
        labs = [lab for lab in labs if lab not in set(exclude_labels)]
        has_raw = all(f"x__{lab}" in data.files for lab in labs)
        phi = np.vstack([data[f"label__{lab}"] for lab in labs])
        labels = np.concatenate([[lab] * len(data[f"label__{lab}"]) for lab in labs])
        x = np.vstack([data[f"x__{lab}"] for lab in labs]) if has_raw else None
        rows = np.concatenate([data[f"row__{lab}"] for lab in labs]) if has_raw else None
        base = list(data["base_feature_names"]) if "base_feature_names" in data.files else None
        return cls(phi, labels, x, rows, list(data["feature_names"]), base,
                   tau_zd if tau_zd is not None else novelty.read_tau(npz_path), source=str(npz_path))

    def __len__(self):
        return len(self.labels)

    # ------------------------------------------------------------------------------------------
    def distances(self, q):
        """Khoang cach L2 tu q toi moi vector cua R (float64)."""
        q = np.asarray(q, dtype="float64").ravel()
        d2 = self._sq - 2.0 * (self.phi @ q) + q @ q
        return np.sqrt(np.maximum(d2, 0.0))

    def lookup(self, q, k_vote=K_VOTE, k_novelty=K_NOVELTY):
        return self._from_distances(self.distances(q), k_vote, k_novelty)

    def lookup_batch(self, Q, chunk=128, **kw):
        """lookup() cho nhieu alert (experiment), tinh khoang cach theo khuc de khong vo RAM."""
        Q = np.asarray(Q, dtype="float64")
        out = []
        for i in range(0, len(Q), chunk):
            q = Q[i:i + chunk]
            d2 = self._sq[None, :] - 2.0 * (q @ self.phi.T) + (q ** 2).sum(axis=1)[:, None]
            out += [self._from_distances(dd, **kw) for dd in np.sqrt(np.maximum(d2, 0.0))]
        return out

    def _from_distances(self, d, k_vote=K_VOTE, k_novelty=K_NOVELTY):
        """Luat vote + novelty tren vector khoang cach d (toi moi vector cua R)."""
        k = max(k_vote, k_novelty)
        nn = np.argpartition(d, k - 1)[:k]
        nn = nn[np.lexsort((nn, d[nn]))]                 # gan nhat truoc; hoa khoang cach -> chi so nho
        vote_idx = nn[:k_vote]
        votes = Counter(self.labels[vote_idx].tolist())
        best = max(votes.values())
        tied = {lab for lab, c in votes.items() if c == best}
        label = next(self.labels[i] for i in vote_idx if self.labels[i] in tied)
        winners = np.array([i for i in vote_idx if self.labels[i] == label])
        s = float(d[nn[:k_novelty]].mean())
        return Lookup(label=str(label), votes=dict(votes), winners=winners, neighbours=nn[:k_novelty],
                      distances=d[nn[:k_novelty]], novelty=s, tau_zd=self.tau_zd,
                      flagged=novelty.is_flagged(s, self.tau_zd))
