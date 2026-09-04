import os
import json
import time

import numpy as np


# Maps each SHAP-centroid label (from RAG_SHAP/kaggle_build_binary_centroids.py,
# trained on Label_Flow/*.csv filenames) to the attack_profiles/*.json "id" that
# holds the full context returned to the VLM. "Benign" maps to attack_profiles/
# benign.json so a benign-looking flow can surface as a real match instead of
# always falling back to unknown_anomaly (see novelty.md gate experiment notes,
# 2026-08-16 — the RAG previously had no way to say "this isn't an attack").
#
# NOTE ON "Infilteration" (2026-08-22): Infiltration is deliberately held out
# of every Layer 1 training split (see binary_split_dataset_z_Infilteration/
# in UTS_Layer1/ and PROJECT_STATE.md) so it can serve as a zero-shot/zero-day
# evaluation case — Layer 1 never sees it during training. The centroid below
# for "Infilteration" is built from held-out data purely for reference/draft
# purposes; it is NOT meant to be a reliable match target, since a system
# that has genuinely never trained on this label should not be expected to
# retrieve it accurately (and empirically doesn't — top-1 retrieval accuracy
# for this label was the weakest of all 15, confused heavily with Benign).
# Don't treat a low/wrong Infilteration match as a retriever bug; it's the
# expected behavior of the zero-shot setup itself.
LABEL_TO_PROFILE_ID = {
    "Benign": "benign",
    "Bot": "bot",
    "Brute Force -Web": "brute_force_web",
    "Brute Force -XSS": "web_attack_xss",
    "DDOS attack-HOIC": "ddos_hoic",
    "DDOS attack-LOIC-UDP": "ddos_udp_flood",
    "DDoS attacks-LOIC-HTTP": "ddos_loic_http",
    "DoS attacks-GoldenEye": "dos_goldeneye",
    "DoS attacks-Hulk": "dos_hulk",
    "DoS attacks-SlowHTTPTest": "dos_slowhttptest",
    "DoS attacks-Slowloris": "dos_slowloris",
    "FTP-BruteForce": "ftp_bruteforce",
    "Infilteration": "infiltration",  # draft/zero-shot only — see note above
    "SQL Injection": "web_attack_sql_injection",
    "SSH-Bruteforce": "ssh_bruteforce",
}


class SHAPCentroidRetriever:
    """RAG-style retriever that matches a flow's SHAP vector directly against
    per-label SHAP centroids (cosine similarity) instead of embedding a text
    query and searching FAISS. See RAG_ISSUES.md for why the text-embedding
    approach (RAGRetriever) fails: SHAP output and natural-language threat
    intel live in different representation spaces, so semantic search on a
    SHAP-derived query string doesn't reliably retrieve the right document
    even when the retrieved KB is correct. Matching raw SHAP vectors against
    centroids computed the same way is exact instead.

    Both the query SHAP vector and the centroids now live in the SAME space:
    the production Layer 1 binary XGBoost-only model's SHAP output (class 1,
    "Anomaly/Malicious"). This class does NOT run its own model or its own
    TreeExplainer pass — it expects the caller (main.py) to have already
    computed full per-flow SHAP vectors via IDSExplainer (the same object
    used to render the waterfall PNGs) and pass them into retrieve_batch().

    This replaces an earlier design (pre-2026-08-22) that used a SEPARATE
    multi-class XGBoost model (RAG_SHAP/train_multiclass_shap_centroid.py)
    trained purely to produce centroids, requiring its own predict + SHAP
    pass on every anomalous flow. That was measured to be more discriminative
    per-flow (71.3% top-1 centroid-retrieval accuracy vs. binary-only's
    69.6%, RAG_SHAP/artifacts/binary_vs_multiclass_shap_view_comparison
    experiments) but paid for it with: (a) a second full TreeExplainer pass
    every time Layer 1 flags an anomaly (latency), and (b) a structural
    mismatch — the multi-class model's SHAP vectors live in a nearly
    orthogonal space from Layer 1's own SHAP output (avg cosine ~0.16 across
    15 labels), so what the retriever "sees" and what the VLM's waterfall
    chart shows could disagree even when both were individually correct.
    Rebuilding centroids from Layer 1's own binary SHAP output
    (RAG_SHAP/kaggle_build_binary_centroids.py, full-scale run: n_eval=7397,
    top-1=69.6%, top-3=94.8%, top-5=98.3% — see
    RAG_SHAP/artifacts/binary_only_centroid_fullscale_summary.json) gave up
    ~1.7pp of top-1 accuracy in exchange for eliminating the second model and
    pass entirely, and for guaranteeing retrieval and VLM evidence always
    agree on what SHAP said, since they now come from the same computation.

    format_context_for_vlm() output is intentionally identical in shape to
    RAGRetriever's, so main.py can swap between them without touching the
    VLM prompt-building code downstream.
    """

    def __init__(self, centroids_path, profiles_dir, threshold=0.8):
        """
        Args:
            centroids_path: path to shap_centroids.json — binary-only centroids
                           built by RAG_SHAP/kaggle_build_binary_centroids.py
                           (core/knowledge_base/embeddings/)
            profiles_dir: attack_profiles/ dir to load full MITRE context from
            threshold: cosine similarity match threshold. Below this for every
                       label -> flagged as a potential zero-day, same convention
                       as RAGRetriever's is_potential_zero_day.
        """
        print("Loading SHAP-centroid components...")
        self.threshold = threshold

        with open(centroids_path, "r", encoding="utf-8") as f:
            centroid_data = json.load(f)
        self.feature_order = centroid_data["feature_order"]
        self.centroids = {
            label: np.array(info["full_mean_shap_vector"], dtype="float32")
            for label, info in centroid_data["centroids"].items()
        }

        self.profiles = {}
        for filename in os.listdir(profiles_dir):
            if filename.endswith(".json"):
                with open(os.path.join(profiles_dir, filename), "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    self.profiles[doc["id"]] = doc
        self.unknown_profile = self.profiles.get("unknown_anomaly")

        print(f"SHAP-Centroid Retriever ready: {len(self.centroids)} centroids, "
              f"{len(self.profiles)} attack profiles, threshold={self.threshold}")

    @staticmethod
    def _cosine_similarity(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    def retrieve_batch(self, full_shap_vectors, top_k=3):
        """Match each flow's already-computed SHAP vector against the
        per-label centroids, returning one retrieval result per flow.

        Args:
            full_shap_vectors: list/array of full-length SHAP vectors (one
                per anomalous flow), in self.feature_order order — this is
                IDSExplainer.explain_and_get_dicts()'s full_vectors return
                value. No model inference or TreeExplainer call happens here;
                this method is pure vector comparison.
            top_k: number of candidate profiles to keep per flow.
        """
        start_time = time.time()
        n = len(full_shap_vectors)

        results = []
        for i in range(n):
            values = np.asarray(full_shap_vectors[i], dtype="float32")

            scores = {label: self._cosine_similarity(values, centroid)
                      for label, centroid in self.centroids.items()}
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
        """Same output shape as RAGRetriever.format_context_for_vlm() so
        main.py can use either retriever interchangeably.

        Uses a neutral "RETRIEVED CONTEXT" header (not "THREAT INTELLIGENCE")
        because the top match may be the benign profile — labeling every
        result as threat intelligence would bias the reader toward assuming
        an attack even when the best match says otherwise.
        """
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
