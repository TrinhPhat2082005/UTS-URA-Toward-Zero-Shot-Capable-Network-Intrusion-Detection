"""
Hieu chinh nguong novelty tau_zd cho reference set Protocol B (paper Eq. zeroday-flag).

CHAY:  python LAYER2/experiments/rq3_retrieval/02_calibrate_novelty.py                     # ae_xgboost + svdd_xgboost
       python LAYER2/experiments/rq3_retrieval/02_calibrate_novelty.py --variants ae_xgboost

  s(q)   = trung binh L2 tu q toi 10 vector gan nhat cua R (SHAPKNNRetriever.novelty_scores)
  tau_zd = Q_0.95 cua s tren alert da biet giu ngoai R: moi alert TP tren val_full o nguong
           van hanh cua Layer 1 (R chi chua flow attack cua train_full).

Ghi vao artifacts/shap_pool_<variant>.meta.json, khoa "novelty" (retriever doc khi tau_zd=None).
Fold LOAO (Setup 2) hieu chinh rieng tren alert da biet cua val_binary tung fold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

paths.use_layer2_code()
import config     # noqa: E402
import pipeline   # noqa: E402
from shap_knn_retriever import SHAPKNNRetriever  # noqa: E402

QUANTILE = 0.95


def known_alerts(l1, thr):
    """SHAP cua moi alert TP tren val_full (chi doc cot can, theo tung khuc)."""
    cols = list(pd.read_csv(config.VAL_BINARY_CSV, nrows=0).columns)
    f = l1.base_feature_columns(pd.DataFrame(columns=cols))
    parts = []
    for ch in pd.read_csv(config.VAL_BINARY_CSV, usecols=f + ["Label"], chunksize=50_000):
        X = ch.loc[ch["Label"] == 1, f].to_numpy("float32")
        if len(X) == 0:
            continue
        X = X[l1.predict_proba(X) >= thr]
        for i in range(0, len(X), 2000):
            parts.append(np.asarray(l1.score(X[i:i + 2000])[1], dtype="float32"))
    return np.vstack(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["ae_xgboost", "svdd_xgboost"])
    args = ap.parse_args()

    for v in args.variants:
        t0 = time.time()
        npz = Path(config.ARTIFACTS_DIR) / f"shap_pool_{v}.npz"
        l1 = pipeline.Layer1Engine(v)
        Q = known_alerts(l1, config.layer1_threshold(v))
        l1 = None
        ret = SHAPKNNRetriever(str(npz), config.ATTACK_PROFILES_DIR, tau_zd=float("inf"),
                               k=config.RAG_NOVELTY_K)
        s = ret.novelty_scores(Q, chunk=16)
        tau = float(np.quantile(s, QUANTILE))

        meta_path = npz.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["novelty"] = {
            "score": f"mean L2 distance to the {config.RAG_NOVELTY_K} nearest reference vectors",
            "k": config.RAG_NOVELTY_K, "quantile": QUANTILE, "tau_zd": tau,
            "calibration_set": "true-positive alerts on val_full at the Layer 1 operating threshold",
            "n_calibration": int(len(s)),
            "score_percentiles": {p: float(np.percentile(s, p)) for p in (50, 90, 95, 99)},
            "calibrated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"{v:<14} tau_zd = {tau:.4f}  (n = {len(s):,}, {time.time() - t0:.0f}s) -> {meta_path.name}",
              flush=True)


if __name__ == "__main__":
    main()
