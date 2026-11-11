"""
Hieu chinh nguong novelty tau_zd cho reference set Protocol B (paper Eq. zeroday-flag).

CHAY:  python LAYER2/experiments/rq3_retrieval/02_calibrate_novelty.py                     # ae_xgboost + svdd_xgboost
       python LAYER2/experiments/rq3_retrieval/02_calibrate_novelty.py --variants ae_xgboost

  s(q)   = trung binh L2 tu q toi 10 vector gan nhat cua R (SHAPKNNRetriever.novelty_scores)
  tau_zd = Q_0.95 cua s tren alert da biet giu ngoai R: moi alert TP tren val_full o nguong
           van hanh cua Layer 1 (R chi chua flow attack cua train_full).

Ghi vao meta cua tap R (reference_sets/protocol_b/<variant>.meta.json), khoa "novelty"
(retriever doc khi tau_zd=None; quy tac co o retrieval/novelty.py).
Fold LOAO (Setup 2) hieu chinh rieng tren alert da biet cua val_binary tung fold
(reference_sets/protocol_a/<fold>/<variant>.meta.json).
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
from gate.engine import Layer1Engine            # noqa: E402
from retrieval import novelty, reference_set      # noqa: E402
from retrieval.retriever import SHAPKNNRetriever  # noqa: E402

QUANTILE = novelty.QUANTILE


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


def calibrate_loao(variant, alpha):
    """Protocol A (Setup 2): tau_zd rieng tung fold.

    Alert da biet cua fold = flow attack (13 loai giu lai) cua val_binary co p >= tau_alpha
    cua fold; s(q) cua chung da duoc rq3b_novelty_build tinh tren chinh pool cua fold
    (output/rq3b_novelty/r3_full/<variant>__<fold>.npz), nen chi can lay phan vi.
    Cung quy uoc voi Detect@5% trong tab:res-novelty (np.quantile mac dinh).
    """
    for fold, npz in reference_set.protocol_a_folds(variant):
        meta_path = novelty.meta_path(npz)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        thr = float(meta["layer1_thresholds_by_alpha"][alpha])
        D = np.load(paths.OUT_RQ3B / "r3_full" / f"{variant}__{fold}.npz", allow_pickle=True)
        j = [f"{b:.3f}" for b in D["thr_budgets"]].index(alpha)
        assert abs(float(D["thr_values"][j]) - thr) < 1e-6, f"{fold}: nguong r3_full != nguong pool"
        s = D["s_full_known"][D["p_known"] >= thr]
        tau = novelty.calibrate_tau(s, QUANTILE)
        meta["novelty"] = {
            "score": f"mean L2 distance to the {config.RAG_NOVELTY_K} nearest reference vectors",
            "k": config.RAG_NOVELTY_K, "quantile": QUANTILE, "tau_zd": tau, "alpha": float(alpha),
            "calibration_set": "alerts raised at the fold's Layer 1 threshold on the attack flows "
                               "of val_binary (retained types only)",
            "source": f"rq3b_novelty_build: r3_full/{variant}__{fold}.npz (s_full_known)",
            "n_calibration": int(len(s)),
            "calibrated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"{fold:<26} tau_alpha = {thr:.4f}  tau_zd = {tau:.4f}  (n = {len(s):,})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["A", "B"], default="B")
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--alpha", default="0.004", help="Protocol A: ngan sach FPR cua Layer 1")
    args = ap.parse_args()

    if args.protocol == "A":
        for v in args.variants or ["ae_xgboost_mono"]:
            calibrate_loao(v, args.alpha)
        return

    for v in args.variants or ["ae_xgboost", "svdd_xgboost"]:
        t0 = time.time()
        npz = reference_set.protocol_b_path(v)
        l1 = Layer1Engine(v)
        Q = known_alerts(l1, config.layer1_threshold(v))
        l1 = None
        ret = SHAPKNNRetriever(str(npz), config.ATTACK_PROFILES_DIR, tau_zd=float("inf"),
                               k=config.RAG_NOVELTY_K)
        s = ret.novelty_scores(Q, chunk=16)
        tau = novelty.calibrate_tau(s, QUANTILE)

        meta_path = novelty.meta_path(npz)
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
