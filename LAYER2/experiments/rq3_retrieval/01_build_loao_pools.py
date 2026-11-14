"""
Build tap tham chieu R cho TUNG FOLD leave-one-attack-out (Protocol A) — dung cho R3 va Setup 2 (VLM zero-day).

CHAY:  python LAYER2/experiments/rq3_retrieval/01_build_loao_pools.py               # ca 3 bien the, 14 fold
       python LAYER2/experiments/rq3_retrieval/01_build_loao_pools.py --variants ae_xgboost_mono

Moi fold f (giu lai nhan h):
  R    = SHAP cua TOAN BO flow attack trong 03_splits_v2/by_label/<h>/train_benign_attack.csv
         (13 loai giu lai, KHONG benign, KHONG gioi han so flow) — cung cach build tap R Protocol B
         (retrieval/reference_set.py).
  SHAP = Layer1Engine.from_fold (gate/engine.py): TreeSHAP chinh xac cua checkpoint Layer 1 cua chinh
         fold do, tren khong gian input day du cua bien the (78 goc + augmented).

Output (cung dinh dang tap R Protocol B, retrieval/store.ReferenceStore.load nap truc tiep):
  <UTS_DATA>/artifacts/reference_sets/protocol_a/<fold>/<variant>.npz + .meta.json
  (meta co them layer1_thresholds_by_alpha: tau_alpha cua fold tren benign train)

Checkpoint lay tu zeroshot_benchmark/04_training_v2/artifacts (ban Layer 1 cuoi: *_mono, xgboost_only).
Chi doc checkpoint + CSV. Bo qua file da ton tai (chay lai an toan).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

paths.use_layer2_code()
from gate.engine import Layer1Engine   # noqa: E402
from retrieval import reference_set      # noqa: E402

SPL = paths.SPLITS
VARIANTS = ["ae_xgboost_mono", "svdd_xgboost_mono", "xgboost_only"]
ALPHAS = [0.001, 0.002, 0.004, 0.008, 0.010, 0.015, 0.020, 0.030]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    args = ap.parse_args()

    folds = sorted(p.name for p in SPL.iterdir() if p.is_dir())
    for fold in folds:
        todo = [v for v in args.variants if not reference_set.protocol_a_path(fold, v).exists()]
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
        X = atk[base].to_numpy("float32")
        y = atk["Label_Text"].to_numpy()
        Xb = ben[base].to_numpy("float32")

        for var in todo:
            l1 = Layer1Engine.from_fold(var, fold, paths.L1_ART, base)
            phi = l1.shap_values(l1.build_input(X)).astype("float32")
            assert phi.shape[1] == len(l1.feature_names), f"{var}/{fold}: {phi.shape[1]} chieu != {len(l1.feature_names)} ten"

            # nguong Layer 1 cua fold theo ngan sach FPR tren benign train (giong danh gia Layer 1)
            p_ben = l1.predict_proba(Xb)
            thr = {f"{a:.3f}": float(np.quantile(p_ben, 1 - a, method="higher")) for a in ALPHAS}

            out = reference_set.save(
                reference_set.protocol_a_path(fold, var),
                {lab: phi[y == lab] for lab in sorted(set(y))}, l1.feature_names, {
                    "fold_held_out": fold, "variant": var,
                    "source_csv": f"03_splits_v2/by_label/{fold}/train_benign_attack.csv",
                    "checkpoint": f"zeroshot_benchmark/04_training_v2/artifacts/{var}/{fold}/checkpoint/model.pkl",
                    "shap": "TreeSHAP (shap.TreeExplainer = xgboost pred_contribs, path-dependent), bias column dropped",
                    "n_dims": len(l1.feature_names), "n_base_features": len(base),
                    "excluded_labels": ["Benign", fold], "cap_per_label": None,
                    "n_vectors_total": int(len(phi)),
                    "labels": {lab: int((y == lab).sum()) for lab in sorted(set(y))},
                    "layer1_thresholds_by_alpha": thr,
                    "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                })
            print(f"{fold:<26} {var:<18} {len(phi):>7,} vector x {len(l1.feature_names)} chieu  "
                  f"-> {out.stat().st_size / 1e6:.1f} MB", flush=True)
            if l1.device.type == "cuda":
                l1.torch.cuda.empty_cache()
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
