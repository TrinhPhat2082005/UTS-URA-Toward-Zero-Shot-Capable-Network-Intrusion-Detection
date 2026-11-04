"""
SHAP-vector retriever (paper Section 3.5). One Euclidean distance pass from the
query attribution vector q to every vector of the reference set R gives both
outputs of the retriever:

  Identification  d_l(q) = min_{r in R_l} ||q - r||_2 for every attack type l;
                  the three types with the smallest d_l form the candidate set
                  C_3(q), ranked; the first one is the retriever's prediction.
  Novelty         s(q) = mean distance to the k = 10 nearest vectors of R;
                  flag_zd = s(q) > tau_zd (retrieval/novelty.py).

All three candidates come from the same rule, so the VLM always receives three
profiles of the same kind, each with its distance. The retriever returns numbers
and profile documents only; the text shown to the VLM is written by prompts.py.

Reference set: retrieval/reference_set.py (label__<type> -> n x d float32), built
from ALL attack flows of the training split, no benign, no cap.
tau_zd is read from the set's .meta.json ("novelty" -> "tau_zd") unless given.
"""
import time

import numpy as np

from knowledge_base.profiles import LABEL_TO_PROFILE_ID, load_profiles

from . import novelty


class SHAPKNNRetriever:
    """Per-type nearest-L2 identification + kNN-10 L2 novelty in one pass."""

    def __init__(self, npz_path, profiles_dir, tau_zd=None, k=10, exclude_labels=None):
        """
        Args:
            npz_path: reference set R (retrieval/reference_set.py).
            profiles_dir: attack_profiles/ dir (one JSON per attack type +
                unknown_anomaly.json).
            tau_zd: novelty threshold. None -> read from <npz>.meta.json; if the
                meta has no calibrated value, the flag is never raised.
            k: neighbors averaged by the novelty score (paper: 10).
            exclude_labels: label names dropped from the reference set.
        """
        print("Loading SHAP-kNN components...")
        self.k = k
        exclude_labels = set(exclude_labels or [])

        data = np.load(npz_path, allow_pickle=True)
        label_keys = [key for key in data.keys() if key.startswith("label__")]

        labels, blocks = [], []
        for key in label_keys:
            label = key[len("label__"):]
            if label in exclude_labels:
                continue
            labels.append(label)
            blocks.append(np.asarray(data[key], dtype="float32"))

        # Xep R thanh tung khoi lien tiep theo nhan: min theo loai = reduceat
        self.labels = np.array(labels)
        self.sizes = np.array([len(b) for b in blocks])
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.pool = np.vstack(blocks)
        self.pool_sq = (self.pool ** 2).sum(axis=1)

        if tau_zd is None:
            tau_zd = novelty.read_tau(npz_path)
        self.tau_zd = tau_zd

        self.profiles = load_profiles(profiles_dir)
        self.unknown_profile = self.profiles.get("unknown_anomaly")

        print(f"SHAP-kNN Retriever ready: {len(self.labels)} types, "
              f"{len(self.pool):,} reference vectors, {len(self.profiles)} profiles, "
              f"k={self.k}, tau_zd={self.tau_zd}")

    def _sq_distances(self, Q):
        """Squared L2 distances, shape (len(Q), len(pool))."""
        d = self.pool_sq[None, :] - 2.0 * (Q @ self.pool.T) + (Q ** 2).sum(axis=1)[:, None]
        return np.maximum(d, 0.0)

    def score(self, Q, top_k=3, chunk=64):
        """Raw retrieval output for a batch of query vectors.

        Returns:
            cand_idx: (n, top_k) indices into self.labels, nearest type first.
            cand_dist: (n, top_k) per-type nearest L2 distance d_l(q).
            novelty: (n,) mean L2 distance to the k nearest reference vectors.
        """
        Q = np.atleast_2d(np.asarray(Q, dtype="float32"))
        cand_idx, cand_dist, s = [], [], []
        for i in range(0, len(Q), chunk):
            d = self._sq_distances(Q[i:i + chunk])
            per_type = np.minimum.reduceat(d, self.starts, axis=1)
            order = np.argsort(per_type, axis=1)[:, :top_k]
            cand_idx.append(order)
            cand_dist.append(np.sqrt(np.take_along_axis(per_type, order, axis=1)))
            nn = np.partition(d, self.k - 1, axis=1)[:, :self.k]
            s.append(np.sqrt(nn).mean(axis=1))
        return np.vstack(cand_idx), np.vstack(cand_dist), np.concatenate(s)

    def novelty_scores(self, Q, chunk=64):
        """s(q) only; used to calibrate tau_zd on held-out known alerts."""
        return self.score(Q, top_k=1, chunk=chunk)[2]

    def _doc_for(self, label):
        profile_id = LABEL_TO_PROFILE_ID.get(label)
        doc = self.profiles.get(profile_id) if profile_id else None
        return doc if doc is not None else self.unknown_profile

    def retrieve_batch(self, full_shap_vectors, top_k=3):
        """One retrieval result per query vector."""
        start_time = time.perf_counter()
        cand_idx, cand_dist, s = self.score(full_shap_vectors, top_k=top_k)
        n = len(s)
        latency_ms = (time.perf_counter() - start_time) * 1000 / max(n, 1)

        results = []
        for i in range(n):
            matches = []
            for j, dist in zip(cand_idx[i], cand_dist[i]):
                label = str(self.labels[j])
                matches.append({"doc": self._doc_for(label), "distance": float(dist),
                                "matched_label": label})
            results.append({
                "predicted_label": matches[0]["matched_label"],
                "matches": matches,
                "novelty_score": float(s[i]),
                "tau_zd": self.tau_zd,
                "is_potential_zero_day": novelty.is_flagged(float(s[i]), self.tau_zd),
                "latency_ms": latency_ms,
            })
        return results
