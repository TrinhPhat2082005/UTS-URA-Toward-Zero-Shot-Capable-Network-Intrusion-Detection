"""
Builds shap_full_xgboost_only.npz: the flat SHAP reference set used by
SHAPKNNRetriever (core/layer2_vlm/shap_knn_retriever.py). Re-run this whenever
the Layer 1 xgboost-only model is retrained on a different data split, so the
retriever's reference set always matches the model currently in
core/layer1_ml/artifacts/xgboost_only/.

Reads a single CSV with a Label_Text column (original attack-type name,
e.g. "Bot", "DDOS_attack_HOIC") alongside the BASE_MODEL_FEATURES columns.
DATA_CSV below is a placeholder path — point it at whichever full-label CSV
you want to build the reference set from before running.

Usage:
    python build_shap_reference.py
"""
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "core", "layer1_ml"))

# Placeholder: point this at a full-label CSV (all attack types present, not
# a leave-one-attack-out split) with a Label_Text column. See
# model_handler.BASE_MODEL_FEATURES for the required feature columns.
DATA_CSV = os.path.join(PROJECT_ROOT, "data", "train_benign_attack_full.csv")

OUT_PATH = os.path.join(PROJECT_ROOT, "core", "knowledge_base", "embeddings", "shap_full_xgboost_only.npz")
CAP_PER_LABEL = 20000
SEED = 42


def main():
    import shap
    from model_handler import Layer1Inference, BASE_MODEL_FEATURES

    l1 = Layer1Inference(my_ip="0.0.0.0", variant="xgboost")
    # Raw TreeExplainer, not IDSExplainer.explain_and_get_dicts() — that method
    # also renders + saves a waterfall PNG per row, which is fine for a
    # handful of live alerts but not for building a reference set out of tens
    # of thousands of rows.
    tree_explainer = shap.TreeExplainer(l1.bst)

    df = pd.read_csv(DATA_CSV)
    if "Label_Text" not in df.columns:
        raise ValueError(f"{DATA_CSV} has no Label_Text column (original attack-type name)")

    npz_data = {"feature_names": np.array(BASE_MODEL_FEATURES, dtype=object)}
    for label, group in df.groupby("Label_Text"):
        if len(group) > CAP_PER_LABEL:
            group = group.sample(n=CAP_PER_LABEL, random_state=SEED)
        group = group.copy()

        l1.run_inference(group)  # scales BASE_MODEL_FEATURES in place
        shap_exp = tree_explainer(group[l1.model_features].astype("float32"))
        # Always the class-1 (attack) SHAP row: the reference set answers
        # "what attack type does this SHAP signature resemble", independent
        # of whether Layer 1 itself predicted attack or benign for this row.
        values = shap_exp.values[:, :, 1] if shap_exp.values.ndim == 3 else shap_exp.values

        key = f"label__{label.replace(' ', '_').replace('/', '-')}"
        npz_data[key] = values.astype("float32")
        print(f"[{label}] n={len(group)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(OUT_PATH, **npz_data)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
