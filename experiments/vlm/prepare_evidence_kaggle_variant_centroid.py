import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model_handler import Layer1Inference, BASE_MODEL_FEATURES  # noqa: E402
from explainer import IDSExplainer  # noqa: E402
from flow_visualizer import generate_flow_charts  # noqa: E402
from shap_centroid_retriever import SHAPCentroidRetriever  # noqa: E402


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


EVIDENCE_DIR = None  # set from args in main()


def build_evidence_for_set(handler, explainer, retriever, df, set_type, compute_aug_fn, feature_names):
    df = df.reset_index(drop=True)
    manifest = []
    predictions = [1] * len(df)

    for i in range(len(df)):
        row = df.iloc[[i]].copy()
        row_dir = os.path.join(EVIDENCE_DIR, set_type, str(i))
        os.makedirs(row_dir, exist_ok=True)

        X_raw = row[BASE_MODEL_FEATURES].astype("float32").values
        X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)
        X_aug = compute_aug_fn(X_raw)
        aug_df = pd.DataFrame(X_aug, columns=feature_names, index=row.index)
        # carry metadata columns through (Peer_IP used by explainer for waterfall title)
        for c in handler.metadata_columns:
            if c in row.columns:
                aug_df[c] = row[c].values

        shap_paths, shap_dicts, full_vectors = explainer.explain_and_get_dicts(
            aug_df, predictions[i:i+1]
        )
        shap_src = shap_paths[0] if shap_paths else None
        shap_dst = None
        if shap_src and os.path.exists(shap_src):
            shap_dst = os.path.join(row_dir, "shap.png")
            os.replace(shap_src, shap_dst)

        chart_paths = generate_flow_charts(df.iloc[i], row_dir)

        rag_result = retriever.retrieve_batch(full_vectors)
        rag_context = retriever.format_context_for_vlm(rag_result[0])

        meta = {
            "row_idx": i,
            "set_type": set_type,
            "label": int(df.iloc[i]["Label"]),
            "xgb_proba": float(df.iloc[i]["xgb_proba"]),
            "source_split": df.iloc[i].get("source_split", None),
            "dst_port": df.iloc[i].get("Dst Port", None),
            "protocol": df.iloc[i].get("Protocol", None),
            "true_attack_label": df.iloc[i].get("true_attack_label", None),
            "shap_top_features": shap_dicts[0] if shap_dicts else {},
            "shap_image": os.path.basename(shap_dst) if shap_dst else None,
            "chart_images": [os.path.basename(p) for p in chart_paths],
            "rag_context": rag_context,
            "rag_predicted_label": rag_result[0]["predicted_label"],
            "rag_is_potential_zero_day": rag_result[0]["is_potential_zero_day"],
        }
        with open(os.path.join(row_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        manifest.append(meta)

        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            print(f"[{set_type}] {i+1}/{len(df)} evidence bundles built", flush=True)

    return manifest


def main():
    global EVIDENCE_DIR
    parser = argparse.ArgumentParser(description="Build SHAP+chart+RAG evidence for any of the 3 Layer 1 variants")
    parser.add_argument("--variant", required=True, choices=["xgboost_only", "ae_augmented", "svdd_augmented"])
    parser.add_argument("--fp_csv", default=os.path.join(SCRIPT_DIR, "fp_set_xgboost_dummy1.csv"))
    parser.add_argument("--tp_csv", required=True)
    parser.add_argument("--evidence_dir", required=True)
    parser.add_argument("--xgb_only_artifacts_dir", default=os.path.join(SCRIPT_DIR, "layer1_artifacts"))
    parser.add_argument("--variants_dir", default=os.path.join(SCRIPT_DIR, "layer1_variants"))
    parser.add_argument("--centroids_path", required=True,
                         help="shap_centroids_{variant}.json path for this variant")
    parser.add_argument("--profiles_dir", default=os.path.join(SCRIPT_DIR, "attack_profiles"))
    parser.add_argument("--exclude_benign_centroid", action="store_true")
    parser.add_argument("--exclude_labels", default="")
    parser.add_argument("--clip_val", type=float, default=20.0)
    args = parser.parse_args()

    EVIDENCE_DIR = os.path.join(SCRIPT_DIR, args.evidence_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    handler = Layer1Inference.__new__(Layer1Inference)
    handler.my_ip = "0.0.0.0"
    handler.device = "cpu"
    handler.metadata_columns = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp', 'Peer_IP']
    handler.threshold = 0.7

    if args.variant == "xgboost_only":
        handler.variant = "xgboost"
        handler.scaler = joblib.load(os.path.join(args.xgb_only_artifacts_dir, "scaler_xgb_only_v1_infilteration.pkl"))
        handler.bst = joblib.load(os.path.join(args.xgb_only_artifacts_dir, "xgb_only_model_v1_infilteration.pkl"))
        feature_names = list(BASE_MODEL_FEATURES)
        handler.model_features = feature_names

        def compute_aug(X_raw):
            return handler.scaler.transform(X_raw)

    elif args.variant == "ae_augmented":
        ae_dir = os.path.join(args.variants_dir, "ae")
        handler.variant = "ae"
        handler.bst = joblib.load(os.path.join(ae_dir, "xgb_aug_model.pkl"))
        handler.scaler = joblib.load(os.path.join(ae_dir, "scaler_ae.pkl"))
        ae_model = BenignAutoencoder(os.path.join(ae_dir, "ae_best.pt"), input_dim=len(BASE_MODEL_FEATURES),
                                      bottleneck_dim=16, device=device)
        feature_names = list(BASE_MODEL_FEATURES) + [f"ae_latent_{i}" for i in range(16)] + ["ae_recon_error"]
        handler.model_features = feature_names

        def compute_aug(X_raw):
            X_s = handler.scaler.transform(X_raw)
            X_s = np.clip(X_s, -args.clip_val, args.clip_val)
            latent, recon_error = ae_model.encode_and_score(X_s)
            return np.concatenate([X_s, latent, recon_error.reshape(-1, 1)], axis=1)

    else:  # svdd_augmented
        svdd_dir = os.path.join(args.variants_dir, "svdd")
        handler.variant = "svdd"
        handler.bst = joblib.load(os.path.join(svdd_dir, "xgb_svdd_model.pkl"))
        handler.scaler = joblib.load(os.path.join(svdd_dir, "scaler_svdd.pkl"))
        svdd_model = MixtureSVDDInference(svdd_dir, input_dim=len(BASE_MODEL_FEATURES), device=device)
        feature_names = list(BASE_MODEL_FEATURES) + [f"svdd_dist_{k}" for k in range(svdd_model.K)] + ["svdd_dmin"]
        handler.model_features = feature_names

        def compute_aug(X_raw):
            X_s = handler.scaler.transform(X_raw)
            X_s = np.clip(X_s, -args.clip_val, args.clip_val)
            sphere_dists = svdd_model.all_sphere_distances(X_s)
            d_min = sphere_dists.min(axis=1)
            return np.concatenate([X_s, sphere_dists, d_min.reshape(-1, 1)], axis=1)

    handler.all_columns = handler.model_features + handler.metadata_columns
    explainer = IDSExplainer(handler.bst, feature_names)

    retriever = SHAPCentroidRetriever(
        centroids_path=args.centroids_path,
        profiles_dir=args.profiles_dir,
    )
    if args.exclude_benign_centroid:
        retriever.centroids.pop("Benign", None)
        print("[A/B] Excluded 'Benign' centroid.", flush=True)
    for label in [l.strip() for l in args.exclude_labels.split(",") if l.strip()]:
        if retriever.centroids.pop(label, None) is not None:
            print(f"[A/B] Excluded '{label}' centroid.", flush=True)
        else:
            print(f"[A/B] WARNING: '{label}' not found among centroids.", flush=True)

    fp_df = pd.read_csv(args.fp_csv)
    tp_df = pd.read_csv(args.tp_csv)
    print(f"[Data] variant={args.variant}  FP rows: {len(fp_df)}, TP rows: {len(tp_df)}", flush=True)

    full_manifest = {}
    full_manifest["fp"] = build_evidence_for_set(handler, explainer, retriever, fp_df, "fp", compute_aug, feature_names)
    full_manifest["tp"] = build_evidence_for_set(handler, explainer, retriever, tp_df, "tp", compute_aug, feature_names)

    manifest_path = os.path.join(EVIDENCE_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(full_manifest, f, indent=2)

    print(f"\nBuilt {len(full_manifest['fp'])} FP + {len(full_manifest['tp'])} TP evidence bundles.", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
