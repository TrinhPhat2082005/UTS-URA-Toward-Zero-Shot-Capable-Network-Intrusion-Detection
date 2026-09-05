"""
Standalone demo: CSV sample -> Layer1 (XGBoost) -> SHAP -> RAG -> MITRE ATT&CK pattern.

Input:  a CICFlowMeter-schema CSV with a 'Label' column (e.g. test_binary.csv)
Output: for each sampled attack flow, the retrieved MITRE ATT&CK technique(s)
        with confidence, tactic, and mitigation.

Usage:
    python test_ml_shap_rag.py --csv path/to/test_binary.csv --n 3
"""
import os
import sys
import argparse

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "core", "layer1_ml"))
sys.path.append(os.path.join(PROJECT_ROOT, "core", "layer2_vlm"))

import ssl
try:
    import certifi
    def _load_certifi_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
        self.load_verify_locations(certifi.where())
    ssl.SSLContext.load_default_certs = _load_certifi_certs
except ImportError:
    pass

from model_handler import Layer1Inference, BASE_MODEL_FEATURES
from explainer import IDSExplainer
from rag_retriever import RAGRetriever


def build_plain_query(shap_dict, raw_values):
    """Experimental alt query: just 'name = value' pairs, no high/low wording,
    to test whether more raw feature context alone (without direction wording)
    helps FAISS retrieval more than the default _build_query() template."""
    parts = []
    for feature in shap_dict:
        value = raw_values.get(feature, "?")
        parts.append(f"{feature} = {value}")
    return "Network flow with: " + ", ".join(parts)


def load_attack_samples(csv_path, n_samples, seed):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "Label" in df.columns:
        # Label may be binary (0/1) or a string category (e.g. "Benign" vs attack name).
        if pd.api.types.is_numeric_dtype(df["Label"]):
            attacks = df[df["Label"] == 1].reset_index(drop=True)
        else:
            attacks = df[df["Label"].str.lower() != "benign"].reset_index(drop=True)
        if attacks.empty:
            raise ValueError(f"No attack rows found in {csv_path} (all rows are benign/Label==0)")
    else:
        # Pre-split-by-attack-type file (e.g. Label_Flow/SSH-Bruteforce.csv):
        # every row is already an attack of that type.
        print("[Input] No 'Label' column found — treating all rows as attack flows.")
        attacks = df

    sample = attacks.sample(n=min(n_samples, len(attacks)), random_state=seed).reset_index(drop=True)

    if "Peer_IP" not in sample.columns:
        sample["Peer_IP"] = sample.get("Dst IP", "unknown")

    for col in BASE_MODEL_FEATURES:
        if col not in sample.columns:
            sample[col] = 0

    return sample


def main():
    parser = argparse.ArgumentParser(description="CSV sample -> Layer1 -> SHAP -> RAG -> MITRE ATT&CK pattern")
    parser.add_argument("--csv", type=str,
                         default=os.path.join(PROJECT_ROOT, "..", "UTS_Layer1",
                                               "binary_split_dataset_z_Infilteration", "test_binary.csv"),
                         help="Path to a CICFlowMeter-schema CSV with a Label column")
    parser.add_argument("--n", type=int, default=3, help="Number of attack flows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--layer1_variant", type=str, choices=["xgboost", "ae", "svdd"], default="xgboost")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="Placeholder host IP (unused for CSV input)")
    parser.add_argument("--top_k_shap", type=int, default=5,
                         help="Number of top SHAP features to use when building the RAG query")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    print(f"[Input] Loading attack samples from: {csv_path}")
    ids_df = load_attack_samples(csv_path, args.n, args.seed)
    print(f"[Input] Sampled {len(ids_df)} attack flow(s) (Label == 1)")

    print(f"\n[Layer 1] Initializing Layer1Inference (variant={args.layer1_variant})...")
    l1_handler = Layer1Inference(my_ip=args.ip, variant=args.layer1_variant)

    # run_inference overwrites BASE_MODEL_FEATURES in ids_df with scaled values
    # (needed so SHAP explains the same values XGBoost decided on) — keep the
    # original unscaled values so the RAG query can cite real numbers (e.g. "Dst Port 22").
    raw_values_per_row = ids_df[BASE_MODEL_FEATURES].to_dict(orient="records")

    predictions, probs = l1_handler.run_inference(ids_df)
    for i in range(len(ids_df)):
        print(f"  Flow {i+1}: pred={predictions[i]} (prob={probs[i]:.4f})")

    print(f"\n[XAI] Running SHAP TreeExplainer (top_k={args.top_k_shap})...")
    explainer = IDSExplainer(l1_handler.bst, l1_handler.model_features)
    shap_paths, shap_dicts = explainer.explain_and_get_dicts(ids_df, predictions, top_k=args.top_k_shap)

    print("\n[RAG] Loading knowledge base...")
    rag = RAGRetriever(
        index_path=os.path.join(PROJECT_ROOT, "core", "knowledge_base", "embeddings", "faiss.index"),
        docs_path=os.path.join(PROJECT_ROOT, "core", "knowledge_base", "embeddings", "documents.json"),
    )

    print("\n" + "=" * 70)
    print("RESULTS: Flow -> SHAP top features -> MITRE ATT&CK pattern")
    print("=" * 70)

    for i, shap_dict in enumerate(shap_dicts):
        label = "Anomaly/Malicious" if predictions[i] == 1 else "Benign"
        print(f"\n--- Flow {i+1} [{label}, prob={probs[i]:.4f}] ---")
        print(f"Top SHAP features: {shap_dict}")

        rag_result = rag.retrieve(shap_dict, top_k=3, raw_values=raw_values_per_row[i])
        print(f"RAG query (default template): {rag_result['query']}")
        print(f"Zero-day (no confident match): {rag_result['is_potential_zero_day']}")

        for m in rag_result["matches"]:
            doc = m["doc"]
            print(f"  -> {doc['title']} | MITRE: {doc['mitre_id']} ({doc['mitre_tactic']}) "
                  f"| confidence={m['confidence']:.2f}")

        # Experimental: plain "name = value" query (no high/low wording)
        plain_query = build_plain_query(shap_dict, raw_values_per_row[i])
        query_vec = rag.encoder.encode([plain_query], normalize_embeddings=True).astype('float32')
        dist, idx = rag.index.search(query_vec, 3)
        print(f"\nRAG query (plain name=value, top_k_shap={args.top_k_shap}): {plain_query}")
        for d, doc_idx in zip(dist[0], idx[0]):
            doc = rag.documents[doc_idx]
            conf = 1.0 / (1.0 + d)
            print(f"  -> {doc['title']} | MITRE: {doc['mitre_id']} ({doc['mitre_tactic']}) "
                  f"| confidence={conf:.2f}")

        print("\nFormatted context for VLM (default template):")
        print(rag.format_context_for_vlm(rag_result))

    print(f"\n[Done] SHAP waterfall PNGs written to: {explainer.output_path}")


if __name__ == "__main__":
    main()
