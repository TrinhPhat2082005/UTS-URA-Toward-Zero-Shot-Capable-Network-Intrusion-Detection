"""
k-NN variant of shap_centroid_retriever.py: matches a flow's SHAP vector
against the FULL per-label build pool (best/max cosine similarity across all
build vectors for that label) instead of a single mean centroid. Same public
interface (retrieve_batch, format_context_for_vlm) as SHAPCentroidRetriever
so main.py / prepare_evidence_kaggle_variant.py can swap between them without
touching downstream code.

Motivation (2026-08-24): RAG_SHAP/eval_retrieval_no_infilt.py measured k-NN
beats centroid-mean on top1 accuracy for known-attack retrieval, at every
Layer 1 variant (+7.2pp to +23.3pp), while trading off zero-day detection
ability (see RAG_SHAP/artifacts/knn_vs_centroid_20k_summary.md). This module
lets the VLM pipeline actually use k-NN retrieval for a real end-to-end
comparison against the production centroid retriever, on the same evidence-
building/VLM-prompting path — instead of the offline-only eval scripts used
so far.

Build pool source: reuses the ALREADY-COMPUTED shap_full_{variant}.npz files
from RAG_SHAP/kaggle_build_all_variants_shap.py (20K-cap per label, seed=42)
— no retraining, no new TreeExplainer pass. Each variant's build pool is the
SAME data the centroid was averaged from, so centroid vs. k-NN comparisons
on identical evidence are apples-to-apples.
"""
import os
import json
import time

import numpy as np

from shap_centroid_retriever import LABEL_TO_PROFILE_ID  # noqa: E402 - reuse the same mapping


class SHAPKNNRetriever:
    """Same public shape as SHAPCentroidRetriever, but retrieve_batch() scores
    each candidate label by its BEST (max) cosine similarity across the
    label's full build pool, not a single mean vector.
    """

    def __init__(self, npz_path, profiles_dir, threshold=0.8, exclude_labels=None):
        """
        Args:
            npz_path: path to shap_full_{variant}.npz (full per-flow SHAP
                      vectors per label, produced by
                      RAG_SHAP/kaggle_build_all_variants_shap.py).
            profiles_dir: attack_profiles/ dir to load full MITRE context from.
            threshold: cosine similarity match threshold. Below this for every
                       label -> flagged as a potential zero-day.
            exclude_labels: iterable of label names to drop from the build
                             pool entirely (e.g. ["Infilteration"]), same
                             semantics as prepare_evidence_kaggle_variant.py's
                             --exclude_labels.
        """
        print("Loading SHAP-kNN components...")
        self.threshold = threshold
        exclude_labels = set(exclude_labels or [])

        data = np.load(npz_path, allow_pickle=True)
        label_keys = [k for k in data.keys() if k.startswith("label__")]
        # Pool cu luu ten co dau cach duoi dang underscore ("DoS attacks-Hulk"
        # -> "DoS_attacks-Hulk") nen phai doi nguoc lai. Pool moi
        # (LAYER2/build_shap_pool.py) luu dung Label_Text, doi se lam hong.
        # Chi doi khi ten goc KHONG co trong LABEL_TO_PROFILE_ID.
        def _restore(k):
            raw = k[len("label__"):]
            return raw if raw in LABEL_TO_PROFILE_ID else raw.replace("_", " ")

        label_map = {k: _restore(k) for k in label_keys}

        self.build_pools = {}
        for key in label_keys:
            label = label_map[key]
            if label in exclude_labels:
                continue
            self.build_pools[label] = np.asarray(data[key], dtype="float32")

        # pre-normalize once so retrieve_batch only needs dot products
        self._norm_pools = {
            label: vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
            for label, vecs in self.build_pools.items()
        }

        self.profiles = {}
        for filename in os.listdir(profiles_dir):
            if filename.endswith(".json"):
                with open(os.path.join(profiles_dir, filename), "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    self.profiles[doc["id"]] = doc
        self.unknown_profile = self.profiles.get("unknown_anomaly")

        print(f"SHAP-kNN Retriever ready: {len(self.build_pools)} label pools "
              f"(sizes: {[len(v) for v in self.build_pools.values()]}), "
              f"{len(self.profiles)} attack profiles, threshold={self.threshold}")

    def retrieve_batch(self, full_shap_vectors, top_k=3):
        """Match each flow's already-computed SHAP vector against every
        label's full build pool, returning one retrieval result per flow.
        Same return shape as SHAPCentroidRetriever.retrieve_batch().
        """
        start_time = time.time()
        n = len(full_shap_vectors)

        results = []
        for i in range(n):
            values = np.asarray(full_shap_vectors[i], dtype="float32")
            v_norm = values / (np.linalg.norm(values) + 1e-12)

            scores = {}
            for label, pool_norm in self._norm_pools.items():
                sims = pool_norm @ v_norm
                scores[label] = float(sims.max())

            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

            best_label, best_score = ranked[0]
            is_potential_zero_day = best_score < self.threshold

            matches = []
            for label, score in ranked:
                profile_id = LABEL_TO_PROFILE_ID.get(label)
                doc = self.profiles.get(profile_id) if profile_id else None
                if doc is None:
                    doc = self.unknown_profile
                matches.append({
                    "doc": doc,
                    "confidence": score,
                    "matched_label": label,
                })

            results.append({
                "predicted_label": best_label,
                "matches": matches,
                "is_potential_zero_day": is_potential_zero_day,
                "latency_ms": (time.time() - start_time) * 1000 / max(n, 1),
            })

        return results

    def format_context_for_vlm(self, retrieval_result: dict) -> str:
        """Identical formatting to SHAPCentroidRetriever.format_context_for_vlm()."""
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
