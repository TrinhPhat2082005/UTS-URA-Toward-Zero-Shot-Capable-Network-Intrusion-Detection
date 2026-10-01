"""
Build reference set R (pool SHAP) cho TUNG FOLD leave-one-attack-out — dung cho R3 va Setup 2 (VLM zero-day).

CHAY:  python LAYER2/experiments/rq3_retrieval/01_build_loao_pools.py               # ca 3 bien the, 14 fold
       python LAYER2/experiments/rq3_retrieval/01_build_loao_pools.py --variants ae_xgboost_mono

Moi fold f (giu lai nhan h):
  pool = SHAP cua TOAN BO flow attack trong 03_splits_v2/by_label/<h>/train_benign_attack.csv
         (13 loai giu lai, KHONG benign, KHONG gioi han so flow) — khop cach build pool cua pipeline
         (build_shap_pool.py cho Protocol B).
  SHAP = TreeSHAP chinh xac cua checkpoint Layer 1 cua chinh fold do (xgboost pred_contribs, bo cot bias),
         tren khong gian input day du cua bien the (78 goc + augmented).

Output (dinh dang giong shap_pool_<variant>.npz cua Protocol B, SHAPKNNRetriever nap truc tiep):
  LAYER2/artifacts/loao_pools/<fold>/shap_pool_<variant>.npz   key label__<Label_Text> + feature_names
  LAYER2/artifacts/loao_pools/<fold>/shap_pool_<variant>.meta.json

Checkpoint lay tu zeroshot_benchmark/04_training_v2/artifacts (ban Layer 1 cuoi: *_mono, xgboost_only).
Chi doc checkpoint + CSV. Bo qua file da ton tai (chay lai an toan).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

paths.use_layer1_code()
from models.autoencoder import BenignAutoencoder   # noqa: E402
from models.svdd import MixtureSVDD                 # noqa: E402

SPL = paths.SPLITS
ART = paths.L1_ART
OUT = paths.LOAO_POOLS
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VARIANTS = ["ae_xgboost_mono", "svdd_xgboost_mono", "xgboost_only"]
ALPHAS = [0.001, 0.002, 0.004, 0.008, 0.010, 0.015, 0.020, 0.030]
CLIP = 20.0
AE_BOTTLENECK = 16


def load_encoders(n_feat):
    ae = BenignAutoencoder(input_dim=n_feat, bottleneck_dim=AE_BOTTLENECK).to(DEV)
    ae.load_state_dict(torch.load(ART / "ae_only/checkpoint/ae_best.pt", map_location=DEV))
    ae.eval()
    k_in = len(list((ART / "svdd_only/checkpoint/svdd_model").glob("encoder_*.pt")))
    svdd = MixtureSVDD(K=k_in, input_dim=n_feat, embed_dim=64, device=DEV)
    svdd.load(str(ART / "svdd_only/checkpoint/svdd_model"))
    return ae, svdd, k_in


def build_input(var, fold, X, ae, svdd, ae_sc, sv_sc):
    if var == "xgboost_only":
        return joblib.load(ART / var / fold / "checkpoint" / "scaler.pkl").transform(X)
    if var == "ae_xgboost_mono":
        Xs = np.clip(ae_sc.transform(X), -CLIP, CLIP).astype("float32")
        zs, es = [], []
        with torch.no_grad():
            for i in range(0, len(Xs), 8192):
                xb = torch.tensor(Xs[i:i + 8192], device=DEV)
                xh, z = ae(xb)
                zs.append(z.cpu().numpy())
                es.append(((xb - xh) ** 2).mean(1).cpu().numpy())
        return np.concatenate([Xs, np.concatenate(zs), np.concatenate(es)[:, None]], 1)
    Xs = np.clip(sv_sc.transform(X), -CLIP, CLIP)
    d = svdd.all_sphere_distances(Xs)
    return np.concatenate([Xs, d, d.min(1, keepdims=True)], 1)


def feature_names(var, base, k_in):
    if var == "ae_xgboost_mono":
        return base + [f"ae_latent_{i}" for i in range(AE_BOTTLENECK)] + ["ae_recon_error"]
    if var == "svdd_xgboost_mono":
        return base + [f"svdd_sphere_{i}" for i in range(k_in)] + ["svdd_d_min"]
    return list(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    args = ap.parse_args()

    folds = sorted(p.name for p in SPL.iterdir() if p.is_dir())
    ae_sc = joblib.load(ART / "ae_only/checkpoint/scaler.pkl")
    sv_sc = joblib.load(ART / "svdd_only/checkpoint/scaler.pkl")
    ae = svdd = k_in = None

    for fold in folds:
        todo = [v for v in args.variants if not (OUT / fold / f"shap_pool_{v}.npz").exists()]
        if not todo:
            print(f"{fold:<26} da co, bo qua", flush=True)
            continue
        t0 = time.time()
        df = pd.read_csv(SPL / fold / "train_benign_attack.csv")
        base = [c for c in df.columns if c not in ("Label", "Label_Text")]
        atk = df[df["Label"] == 1].reset_index(drop=True)
        ben = df[df["Label"] == 0]
        del df
        assert fold not in set(atk["Label_Text"]), f"{fold}: nhan bi giu lai lot vao pool"
        if ae is None:
            ae, svdd, k_in = load_encoders(len(base))
        X = atk[base].to_numpy("float32")
        y = atk["Label_Text"].to_numpy()
        Xb = ben[base].to_numpy("float32")
        (OUT / fold).mkdir(parents=True, exist_ok=True)

        for var in todo:
            model = joblib.load(ART / var / fold / "checkpoint" / "model.pkl")
            booster = model.get_booster()
            booster.set_param({"device": "cuda" if DEV.type == "cuda" else "cpu"})
            A = build_input(var, fold, X, ae, svdd, ae_sc, sv_sc)
            phi = booster.predict(xgb.DMatrix(A), pred_contribs=True)[:, :-1].astype("float32")
            names = feature_names(var, base, k_in)
            assert phi.shape[1] == len(names), f"{var}/{fold}: {phi.shape[1]} chieu != {len(names)} ten"

            # nguong Layer 1 cua fold theo ngan sach FPR tren benign train (giong danh gia Layer 1)
            p_ben = model.predict_proba(build_input(var, fold, Xb, ae, svdd, ae_sc, sv_sc))[:, 1]
            thr = {f"{a:.3f}": float(np.quantile(p_ben, 1 - a, method="higher")) for a in ALPHAS}

            pools = {f"label__{lab}": phi[y == lab] for lab in sorted(set(y))}
            out = OUT / fold / f"shap_pool_{var}.npz"
            np.savez_compressed(out, feature_names=np.array(names, dtype=object), **pools)
            meta = {
                "fold_held_out": fold, "variant": var,
                "source_csv": f"03_splits_v2/by_label/{fold}/train_benign_attack.csv",
                "checkpoint": f"zeroshot_benchmark/04_training_v2/artifacts/{var}/{fold}/checkpoint/model.pkl",
                "shap": "TreeSHAP (xgboost pred_contribs, path-dependent), bias column dropped",
                "n_dims": len(names), "n_base_features": len(base),
                "excluded_labels": ["Benign", fold], "cap_per_label": None,
                "n_vectors_total": int(len(phi)),
                "labels": {lab: int((y == lab).sum()) for lab in sorted(set(y))},
                "layer1_thresholds_by_alpha": thr,
                "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            }
            out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"{fold:<26} {var:<18} {len(phi):>7,} vector x {len(names)} chieu  "
                  f"-> {out.stat().st_size / 1e6:.1f} MB", flush=True)
            if DEV.type == "cuda":
                torch.cuda.empty_cache()
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
