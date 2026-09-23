import os
import json
import time

import numpy as np

from shap_centroid_retriever import LABEL_TO_PROFILE_ID  # noqa: E402 - reuse the same mapping


class SHAPKNNRetriever:
    """True k-NN over the full flat SHAP reference set (Eq. 8-9,
    03_methodology.tex sec:method:retrieval:formal): cosine similarity to
    every reference vector regardless of label, top-k neighbours, majority
    vote on their labels. Same public interface as SHAPCentroidRetriever.
    """

    def __init__(self, npz_path, profiles_dir, threshold=0.8, exclude_labels=None):
        print("Loading SHAP-kNN components...")
        self.threshold = threshold
        exclude_labels = set(exclude_labels or [])

        data = np.load(npz_path, allow_pickle=True)
        label_keys = [k for k in data.keys() if k.startswith("label__")]
        # npz keys use "_" for spaces (kaggle_build_all_variants_shap.py); undo that.
        label_map = {k: k[len("label__"):].replace("_", " ") for k in label_keys}

        vecs_list = []
        labels_list = []
        pool_sizes = {}
        for key in label_keys:
            label = label_map[key]
            if label in exclude_labels:
                continue
            vecs = np.asarray(data[key], dtype="float32")
            vecs_list.append(vecs)
            labels_list.extend([label] * len(vecs))
            pool_sizes[label] = len(vecs)

        # One flat reference matrix R across all labels, not per-label pools.
        self._reference_matrix = np.concatenate(vecs_list, axis=0)
        self._reference_labels = np.asarray(labels_list)
        norms = np.linalg.norm(self._reference_matrix, axis=1, keepdims=True)
        self._reference_matrix_norm = self._reference_matrix / (norms + 1e-12)

        self.profiles = {}
        for filename in os.listdir(profiles_dir):
            if filename.endswith(".json"):
                with open(os.path.join(profiles_dir, filename), "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    self.profiles[doc["id"]] = doc
        self.unknown_profile = self.profiles.get("unknown_anomaly")

        print(f"SHAP-kNN Retriever ready: {len(self._reference_matrix)} reference "
              f"vectors across {len(pool_sizes)} labels (sizes: {list(pool_sizes.values())}), "
              f"{len(self.profiles)} attack profiles, threshold={self.threshold}")

    def retrieve_batch(self, full_shap_vectors, k=5, top_k=3):
        start_time = time.time()
        n = len(full_shap_vectors)
        k = min(k, len(self._reference_matrix))

        results = []
        for i in range(n):
            values = np.asarray(full_shap_vectors[i], dtype="float32")
            v_norm = values / (np.linalg.norm(values) + 1e-12)

            sims = self._reference_matrix_norm @ v_norm

            # argpartition for the k nearest, then sort just those k by similarity.
            knn_idx = np.argpartition(-sims, k - 1)[:k]
            knn_idx = knn_idx[np.argsort(-sims[knn_idx])]
            knn_sims = sims[knn_idx]
            knn_labels = self._reference_labels[knn_idx]

            mean_sim = float(knn_sims.mean())
            is_potential_zero_day = mean_sim < self.threshold

            # Majority vote; mean similarity per label breaks ties / orders candidates.
            vote_counts = {}
            vote_sim_sum = {}
            for label, sim in zip(knn_labels, knn_sims):
                vote_counts[label] = vote_counts.get(label, 0) + 1
                vote_sim_sum[label] = vote_sim_sum.get(label, 0.0) + float(sim)

            ranked = sorted(
                vote_counts.keys(),
                key=lambda lbl: (vote_counts[lbl], vote_sim_sum[lbl] / vote_counts[lbl]),
                reverse=True,
            )[:top_k]

            best_label = ranked[0]

            matches = []
            for label in ranked:
                profile_id = LABEL_TO_PROFILE_ID.get(label)
                doc = self.profiles.get(profile_id) if profile_id else None
                if doc is None:
                    doc = self.unknown_profile
                matches.append({
                    "doc": doc,
                    "confidence": vote_sim_sum[label] / vote_counts[label],
                    "matched_label": label,
                    "votes": vote_counts[label],
                })

            results.append({
                "predicted_label": best_label,
                "matches": matches,
                "is_potential_zero_day": is_potential_zero_day,
                "latency_ms": (time.time() - start_time) * 1000 / max(n, 1),
            })

        return results

    def format_context_for_vlm(self, retrieval_result: dict) -> str:
        context = "### RETRIEVED CONTEXT (MITRE ATT&CK / BENIGN PROFILES)\n"

        if retrieval_result["is_potential_zero_day"]:
            context += "> [!WARNING]\n"
            context += "> The observed behavior has LOW similarity to known attack patterns. "
            context += "This may be a NOVEL ATTACK or ZERO-DAY exploit.\n\n"

        for i, match in enumerate(retrieval_result["matches"]):
            doc = match["doc"]
            conf = match["confidence"]
            is_benign = doc.get("id") == "benign"
            label = "Benign Pattern" if is_benign else "Technique"

            context += f"#### Match {i+1} (Confidence: {conf:.2f})\n"
            context += f"- **{label}:** {doc['title']} ({doc['mitre_id']})\n"
            context += f"- **Tactic:** {doc['mitre_tactic']}\n"
            context += f"- **Description:** {doc['description']}\n"
            context += f"- **Typical Pattern:** {doc['network_indicators']['traffic_pattern']}\n"
            if not is_benign:
                context += f"- **Mitigation:** {', '.join([m['name'] for m in doc['mitigation']])}\n"
            if doc.get("mitre_url") and doc["mitre_url"] != "N/A":
                context += f"- **Reference:** {doc['mitre_url']}\n"
            context += "\n"

        return context
