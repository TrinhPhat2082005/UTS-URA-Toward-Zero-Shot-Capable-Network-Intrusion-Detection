"""
SHAP-vector retriever (paper Section 3.5). One Euclidean distance pass from the
query attribution vector q to every vector of the reference set R gives both
outputs of the retriever:

  Identification  d_l(q) = min_{r in R_l} ||q - r||_2 for every attack type l;
                  the three types with the smallest d_l form the candidate set
                  C_3(q), ranked; the first one is the retriever's prediction.
  Novelty         s(q) = mean distance to the k = 10 nearest vectors of R;
                  flag_zd = s(q) > tau_zd, tau_zd = 95th percentile of s over
                  known alerts held out from R (experiments/rq3_retrieval/02_calibrate_novelty.py).

All three candidates come from the same rule, so the VLM always receives three
profiles of the same kind, each with its distance. When the flag is raised the
unknown_anomaly profile is added and the candidates are presented only as the
known behaviors the alert is closest to.

Reference set: shap_pool_<variant>.npz (label__<type> -> n x d float32), built
from ALL attack flows of the training split, no benign, no cap
(build_shap_pool.py for Protocol B, experiments/rq3_retrieval/01_build_loao_pools.py per fold).
tau_zd is read from the pool's .meta.json ("novelty" -> "tau_zd") unless given.
"""
import os
import json
import time

import numpy as np


# Label_Text (ten nhan trong LAYER2/data/*.csv va khoa label__<nhan> cua pool)
# -> id cua profile trong knowledge_base/attack_profiles/*.json.
LABEL_TO_PROFILE_ID = {
    "Bot": "bot",
    "Brute_Force_Web": "brute_force_web",
    "Brute_Force_XSS": "web_attack_xss",
    "DDOS_attack_HOIC": "ddos_hoic",
    "DDOS_attack_LOIC_UDP": "ddos_udp_flood",
    "DDoS_attacks_LOIC_HTTP": "ddos_loic_http",
    "DoS_attacks_GoldenEye": "dos_goldeneye",
    "DoS_attacks_Hulk": "dos_hulk",
    "DoS_attacks_SlowHTTPTest": "dos_slowhttptest",
    "DoS_attacks_Slowloris": "dos_slowloris",
    "FTP_BruteForce": "ftp_bruteforce",
    "Infiltration": "infiltration",
    "SQL_Injection": "web_attack_sql_injection",
    "SSH_Bruteforce": "ssh_bruteforce",
}


class SHAPKNNRetriever:
    """Per-type nearest-L2 identification + kNN-10 L2 novelty in one pass."""

    def __init__(self, npz_path, profiles_dir, tau_zd=None, k=10, exclude_labels=None):
        """
        Args:
            npz_path: shap_pool_<variant>.npz.
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
            tau_zd = self._tau_from_meta(npz_path)
        self.tau_zd = tau_zd

        self.profiles = {}
        for filename in os.listdir(profiles_dir):
            if filename.endswith(".json"):
                with open(os.path.join(profiles_dir, filename), "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    self.profiles[doc["id"]] = doc
        self.unknown_profile = self.profiles.get("unknown_anomaly")

        print(f"SHAP-kNN Retriever ready: {len(self.labels)} types, "
              f"{len(self.pool):,} reference vectors, {len(self.profiles)} profiles, "
              f"k={self.k}, tau_zd={self.tau_zd}")

    @staticmethod
    def _tau_from_meta(npz_path):
        meta_path = os.path.splitext(npz_path)[0] + ".meta.json"
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return (meta.get("novelty") or {}).get("tau_zd")

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
        cand_idx, cand_dist, novelty = [], [], []
        for i in range(0, len(Q), chunk):
            d = self._sq_distances(Q[i:i + chunk])
            per_type = np.minimum.reduceat(d, self.starts, axis=1)
            order = np.argsort(per_type, axis=1)[:, :top_k]
            cand_idx.append(order)
            cand_dist.append(np.sqrt(np.take_along_axis(per_type, order, axis=1)))
            nn = np.partition(d, self.k - 1, axis=1)[:, :self.k]
            novelty.append(np.sqrt(nn).mean(axis=1))
        return np.vstack(cand_idx), np.vstack(cand_dist), np.concatenate(novelty)

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
        cand_idx, cand_dist, novelty = self.score(full_shap_vectors, top_k=top_k)
        n = len(novelty)
        latency_ms = (time.perf_counter() - start_time) * 1000 / max(n, 1)

        results = []
        for i in range(n):
            matches = []
            for j, dist in zip(cand_idx[i], cand_dist[i]):
                label = str(self.labels[j])
                matches.append({"doc": self._doc_for(label), "distance": float(dist),
                                "matched_label": label})
            s = float(novelty[i])
            results.append({
                "predicted_label": matches[0]["matched_label"],
                "matches": matches,
                "novelty_score": s,
                "tau_zd": self.tau_zd,
                "is_potential_zero_day": bool(self.tau_zd is not None and s > self.tau_zd),
                "latency_ms": latency_ms,
            })
        return results

    def format_context_for_vlm(self, retrieval_result: dict) -> str:
        """Context block: three candidate profiles, nearest first, each labelled with its
        attack type and distance. Every alert gets the novelty score as a distance
        scale; when the flag is raised, the unknown_anomaly profile is added first
        under the attack type UNKNOWN."""
        flagged = retrieval_result["is_potential_zero_day"]
        context = "### RETRIEVED CONTEXT (MITRE ATT&CK PROFILES)\n"

        if not flagged:
            context += (f"Novelty: mean L2 distance to the {self.k} nearest known attack flows = "
                        f"{retrieval_result['novelty_score']:.3f}"
                        + (f" (threshold {self.tau_zd:.3f})" if self.tau_zd is not None else "") + "\n\n")

        if flagged:
            doc = self.unknown_profile
            context += (f"NOVELTY FLAG RAISED: the mean L2 distance from this flow to its {self.k} "
                        f"nearest known attack flows is {retrieval_result['novelty_score']:.3f}"
                        + (f", above the threshold {self.tau_zd:.3f}" if self.tau_zd is not None else "")
                        + ". The flow lies outside the region occupied by known attack types and "
                        "may be a novel attack.\n\n")
            context += "#### Attack type: UNKNOWN\n"
            context += f"- **Profile:** {doc['title']}\n"
            context += f"- **Description:** {doc['description']}\n"
            context += f"- **Typical Pattern:** {doc['network_indicators']['traffic_pattern']}\n"
            context += f"- **Mitigation:** {', '.join([m['name'] for m in doc['mitigation']])}\n\n"

        context += "Candidates are listed nearest first.\n\n"
        for i, match in enumerate(retrieval_result["matches"]):
            doc = match["doc"]
            context += (f"#### Candidate {i+1} - Attack type: {match['matched_label']} "
                        f"(distance to nearest known flow: {match['distance']:.3f})\n")
            context += f"- **Technique:** {doc['title']} ({doc['mitre_id']})\n"
            context += f"- **Tactic:** {doc['mitre_tactic']}\n"
            context += f"- **Description:** {doc['description']}\n"
            context += f"- **Typical Pattern:** {doc['network_indicators']['traffic_pattern']}\n"
            context += f"- **Mitigation:** {', '.join([m['name'] for m in doc['mitigation']])}\n"
            if doc.get("mitre_url") and doc["mitre_url"] != "N/A":
                context += f"- **Reference:** {doc['mitre_url']}\n"
            context += "\n"

        return context
