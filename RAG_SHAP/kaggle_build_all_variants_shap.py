import os
import json
import argparse

import numpy as np
import pandas as pd
import joblib
import shap
import torch
import torch.nn as nn
# Path trên kaggle cần chỉnh lại path local nếu preproduce
LABEL_DIR = "/kaggle/input/datasets/phttrnh/cicids2018-label/Label_Flow"
VARIANTS_DIR = "/kaggle/working/rag_shap_tune/layer1_variants"
XGB_ONLY_MODEL = "/kaggle/working/vlm_gate/layer1_artifacts/xgb_only_model_v1_infilteration.pkl"
XGB_ONLY_SCALER = "/kaggle/working/vlm_gate/layer1_artifacts/scaler_xgb_only_v1_infilteration.pkl"
OUT_DIR = "/kaggle/working/rag_shap_tune/all_variants_shap"

BASE_MODEL_FEATURES = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
    "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg", "Bwd Blk Rate Avg", "Subflow Fwd Pkts",
    "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts",
    "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max",
    "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


class BenignAutoencoder:
    def __init__(self, checkpoint_path, input_dim, bottleneck_dim=16, device="cpu"):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(32, bottleneck_dim), nn.LayerNorm(bottleneck_dim), nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(bottleneck_dim, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(32, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(64, input_dim),
                )

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z), z

        self.device = device
        self.model = Net().to(device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.eval()

    def encode_and_score(self, X_np):
        with torch.no_grad():
            x = torch.tensor(X_np, dtype=torch.float32, device=self.device)
            x_hat, z = self.model(x)
            err = ((x - x_hat) ** 2).mean(dim=1)
        return z.cpu().numpy(), err.cpu().numpy()


class MixtureSVDDInference:
    def __init__(self, artifacts_dir, input_dim, embed_dim=64, device="cpu"):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(128, embed_dim, bias=False),
                )

            def forward(self, x):
                return nn.functional.normalize(self.net(x), dim=1)

        self.device = device
        centers_np = np.load(os.path.join(artifacts_dir, "centers.npy"), allow_pickle=True)
        self.K = len(centers_np)
        self.encoders = []
        self.centers = []
        for k in range(self.K):
            enc = Encoder().to(device)
            enc.load_state_dict(torch.load(os.path.join(artifacts_dir, f"encoder_{k}.pt"), map_location=device))
            enc.eval()
            self.encoders.append(enc)
            self.centers.append(torch.tensor(centers_np[k], dtype=torch.float32, device=device))

    def all_sphere_distances(self, X_np):
        with torch.no_grad():
            x = torch.tensor(X_np, dtype=torch.float32, device=self.device)
            dists = []
            for k in range(self.K):
                z = self.encoders[k](x)
                dists.append(((z - self.centers[k]) ** 2).sum(dim=1).cpu().numpy())
        return np.stack(dists, axis=1)


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def load_label_csv(csv_path, cap, seed):
    df = pd.read_csv(csv_path, on_bad_lines="skip", dtype=str, low_memory=False)
    df.columns = df.columns.str.strip()
    if "Dst Port" in df.columns:
        df = df[df["Dst Port"] != "Dst Port"].reset_index(drop=True)
    for col in BASE_MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if len(df) > cap:
        df = df.sample(n=cap, random_state=seed).reset_index(drop=True)
    return df


def build_variant(variant_name, feature_names, bst, explainer, compute_X_fn, args):
    """Runs the full per-label SHAP + centroid + top-k pipeline for one
    Layer 1 variant. compute_X_fn(df_raw) -> (X_final, proba) for that variant.
    """
    print(f"\n{'='*90}\n[VARIANT] {variant_name}  (n_features={len(feature_names)})\n{'='*90}")

    label_files = sorted(f for f in os.listdir(LABEL_DIR) if f.endswith(".csv"))
    per_label_shap = {}
    per_label_meta = {}

    for fname in label_files:
        label = fname[:-4]
        csv_path = os.path.join(LABEL_DIR, fname)
        df = load_label_csv(csv_path, args.cap, args.seed)
        if len(df) == 0:
            print(f"[Skip] {label}: no rows")
            continue

        X_final, proba = compute_X_fn(df)
        shap_exp = explainer(X_final)
        values = (shap_exp.values[:, :, 1] if shap_exp.values.ndim == 3 else shap_exp.values)

        per_label_shap[label] = values.astype("float32")
        per_label_meta[label] = {"n": len(df), "mean_proba": float(np.mean(proba))}
        print(f"[SHAP] {variant_name:10s} {label:28s} n={len(df):6d}  mean_proba={np.mean(proba):.3f}")

    #  Save FULL per-flow SHAP matrices (the reusable raw data) 
    npz_path = os.path.join(OUT_DIR, f"shap_full_{variant_name}.npz")
    np.savez_compressed(
        npz_path,
        feature_names=np.array(feature_names, dtype=object),
        **{f"label__{lbl.replace(' ', '_').replace('/', '-')}": arr for lbl, arr in per_label_shap.items()},
    )
    print(f"[Saved] {npz_path} (full per-flow SHAP matrices, all labels)")

    #  70/30 build/eval split, mean centroid from build split only 
    rng = np.random.RandomState(args.seed)
    centroids = {}
    eval_rows = []
    for label, values in per_label_shap.items():
        n = len(values)
        idx = rng.permutation(n)
        n_build = max(1, int(round(n * args.build_frac)))
        build_idx, eval_idx = idx[:n_build], idx[n_build:]
        centroids[label] = values[build_idx].mean(axis=0)
        for i in eval_idx:
            eval_rows.append((label, values[i]))

    label_names = list(centroids.keys())
    top1_hits, top3_hits, top5_hits = 0, 0, 0
    per_label_correct = {lbl: {"n": 0, "top1": 0, "top3": 0, "top5": 0} for lbl in label_names}

    for true_label, vec in eval_rows:
        scores = [(lbl, cosine_similarity(vec, centroids[lbl])) for lbl in label_names]
        scores.sort(key=lambda kv: kv[1], reverse=True)
        ranked_labels = [lbl for lbl, _ in scores]
        per_label_correct[true_label]["n"] += 1
        if ranked_labels[0] == true_label:
            top1_hits += 1
            per_label_correct[true_label]["top1"] += 1
        if true_label in ranked_labels[:3]:
            top3_hits += 1
            per_label_correct[true_label]["top3"] += 1
        if true_label in ranked_labels[:5]:
            top5_hits += 1
            per_label_correct[true_label]["top5"] += 1

    n_eval = len(eval_rows)
    print(f"\n[{variant_name}] OVERALL: n_eval={n_eval}  "
          f"top1={top1_hits/n_eval:.4f}  top3={top3_hits/n_eval:.4f}  top5={top5_hits/n_eval:.4f}")

    rows = []
    for lbl in label_names:
        d = per_label_correct[lbl]
        n = d["n"] if d["n"] else 1
        rows.append({
            "label": lbl,
            "n_build": per_label_meta[lbl]["n"] - d["n"],
            "n_eval": d["n"],
            "mean_proba": round(per_label_meta[lbl]["mean_proba"], 3),
            "top1_acc": round(d["top1"] / n, 3),
            "top3_acc": round(d["top3"] / n, 3),
            "top5_acc": round(d["top5"] / n, 3),
        })
    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(OUT_DIR, f"per_label_{variant_name}.csv"), index=False)

    #  Centroid JSON (mean, from ALL data — reference aggregation only) 
    full_centroids = {
        label: {"full_mean_shap_vector": values.mean(axis=0).tolist(), "n_samples_used": len(values)}
        for label, values in per_label_shap.items()
    }
    centroid_out = {
        "feature_order": feature_names,
        "source_model": f"{variant_name} (production Layer 1 variant)",
        "centroids": full_centroids,
    }
    with open(os.path.join(OUT_DIR, f"shap_centroids_{variant_name}.json"), "w", encoding="utf-8") as f:
        json.dump(centroid_out, f, indent=2)

    summary = {
        "variant": variant_name,
        "n_features": len(feature_names),
        "cap": args.cap, "seed": args.seed, "build_frac": args.build_frac,
        "n_eval": n_eval,
        "top1_acc": top1_hits / n_eval, "top3_acc": top3_hits / n_eval, "top5_acc": top5_hits / n_eval,
    }
    with open(os.path.join(OUT_DIR, f"summary_{variant_name}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build_frac", type=float, default=0.7)
    parser.add_argument("--clip_val", type=float, default=20.0)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    all_summaries = []

    # - Variant 1: xgboost-only -
    bst_xo = joblib.load(XGB_ONLY_MODEL)
    scaler_xo = joblib.load(XGB_ONLY_SCALER)
    explainer_xo = shap.TreeExplainer(bst_xo)

    def compute_X_xgboost(df):
        X_raw = df[BASE_MODEL_FEATURES].astype("float32").values
        X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)
        X_s = scaler_xo.transform(X_raw)
        proba = bst_xo.predict_proba(X_s)[:, 1]
        return X_s, proba

    all_summaries.append(build_variant("xgboost_only", BASE_MODEL_FEATURES, bst_xo, explainer_xo, compute_X_xgboost, args))

    # - Variant 2: AE-augmented -
    ae_dir = os.path.join(VARIANTS_DIR, "ae")
    bst_ae = joblib.load(os.path.join(ae_dir, "xgb_aug_model.pkl"))
    scaler_ae = joblib.load(os.path.join(ae_dir, "scaler_ae.pkl"))
    ae_model = BenignAutoencoder(os.path.join(ae_dir, "ae_best.pt"), input_dim=len(BASE_MODEL_FEATURES),
                                  bottleneck_dim=16, device=device)
    ae_feature_names = list(BASE_MODEL_FEATURES) + [f"ae_latent_{i}" for i in range(16)] + ["ae_recon_error"]
    explainer_ae = shap.TreeExplainer(bst_ae)

    def compute_X_ae(df):
        X_raw = df[BASE_MODEL_FEATURES].astype("float32").values
        X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)
        X_s = scaler_ae.transform(X_raw)
        X_s = np.clip(X_s, -args.clip_val, args.clip_val)
        latent, recon_error = ae_model.encode_and_score(X_s)
        X_aug = np.concatenate([X_s, latent, recon_error.reshape(-1, 1)], axis=1)
        proba = bst_ae.predict_proba(X_aug)[:, 1]
        return X_aug, proba

    all_summaries.append(build_variant("ae_augmented", ae_feature_names, bst_ae, explainer_ae, compute_X_ae, args))

    # - Variant 3: SVDD-augmented -
    svdd_dir = os.path.join(VARIANTS_DIR, "svdd")
    bst_svdd = joblib.load(os.path.join(svdd_dir, "xgb_svdd_model.pkl"))
    scaler_svdd = joblib.load(os.path.join(svdd_dir, "scaler_svdd.pkl"))
    svdd_model = MixtureSVDDInference(svdd_dir, input_dim=len(BASE_MODEL_FEATURES), device=device)
    svdd_feature_names = list(BASE_MODEL_FEATURES) + [f"svdd_dist_{k}" for k in range(svdd_model.K)] + ["svdd_dmin"]
    explainer_svdd = shap.TreeExplainer(bst_svdd)

    def compute_X_svdd(df):
        X_raw = df[BASE_MODEL_FEATURES].astype("float32").values
        X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)
        X_s = scaler_svdd.transform(X_raw)
        X_s = np.clip(X_s, -args.clip_val, args.clip_val)
        sphere_dists = svdd_model.all_sphere_distances(X_s)
        d_min = sphere_dists.min(axis=1)
        X_aug = np.concatenate([X_s, sphere_dists, d_min.reshape(-1, 1)], axis=1)
        proba = bst_svdd.predict_proba(X_aug)[:, 1]
        return X_aug, proba

    all_summaries.append(build_variant("svdd_augmented", svdd_feature_names, bst_svdd, explainer_svdd, compute_X_svdd, args))

    print("\n" + "=" * 90)
    print("CROSS-VARIANT SUMMARY")
    print("=" * 90)
    print(pd.DataFrame(all_summaries).to_string(index=False))
    with open(os.path.join(OUT_DIR, "cross_variant_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n[Saved] {OUT_DIR}/cross_variant_summary.json")


if __name__ == "__main__":
    main()
