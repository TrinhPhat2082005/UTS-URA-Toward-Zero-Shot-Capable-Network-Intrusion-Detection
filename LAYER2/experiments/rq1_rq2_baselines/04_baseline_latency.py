"""Step 4 - per-flow latency of every baseline (same protocol as rq5_latency).

One flow per call, as in deployment; 20 warm-up + 300 measured calls, time.perf_counter,
CPU only (GPU hidden), 4 torch threads; fold-Bot checkpoints, flows from that fold's
val_binary.csv. Reports mean and p95 in ms for
    decision             raw features -> attack score
    decision+treeshap    same, plus exact TreeSHAP of the final tree model (tree models only)

Output: OUT_BASE/baseline_latency.json
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import baselines as B  # noqa: E402

N_WARM, N_MEAS, FOLD = 20, 300, "Bot"
OUT = paths.ensure(paths.OUT_BASE)
torch.set_num_threads(4)


def bench(fn, rows):
    ts = []
    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        fn(r)
        if i >= N_WARM:
            ts.append((time.perf_counter() - t0) * 1000)
    ts = np.asarray(ts)
    return {"mean_ms": float(ts.mean()), "p95_ms": float(np.percentile(ts, 95)), "n": int(len(ts))}


def tree_parts(name):
    """(final tree model, raw X -> model input) for the exact-TreeSHAP latency."""
    ck = B._ck(name, FOLD)
    if name in ("xgboost_only", "lgbm"):
        return joblib.load(ck / "model.pkl"), joblib.load(ck / "scaler.pkl").transform
    if name == "random_forest":
        return joblib.load(ck / "model_sklearn_cpu.pkl"), (lambda X: X)
    if name == "ae_xgboost":
        fwd = B._ae()

        def ae_feats(X):
            Xs, z, e = fwd(X)
            return np.concatenate([Xs, z, e[:, None]], 1)
        return joblib.load(ck / "model.pkl"), ae_feats
    if name == "svdd_xgboost":
        fwd = B._svdd()

        def svdd_feats(X):
            Xs, d = fwd(X)
            return np.concatenate([Xs, d, d.min(1, keepdims=True)], 1)
        return joblib.load(ck / "model.pkl"), svdd_feats
    if name == "vae_xgboost":
        fwd = B._vae()

        def vae_feats(X):
            Xs, mu, s = fwd(X)
            return np.concatenate([Xs, mu, s[:, None]], 1)
        return joblib.load(ck / "model.pkl"), vae_feats
    return None, None


def main(names):
    va = pd.read_csv(paths.SPLITS / FOLD / "val_binary.csv")
    feats = [c for c in va.columns if c not in ("Label", "Label_Text")]
    idx = np.random.default_rng(0).choice(len(va), N_WARM + N_MEAS, replace=False)
    X = va[feats].to_numpy("float32")[idx]
    rows = [X[i:i + 1] for i in range(len(X))]
    out_f = OUT / "baseline_latency.json"
    res = json.loads(out_f.read_text(encoding="utf-8")) if names and out_f.exists() else {}
    res.update({"cpu": platform.processor(), "torch_threads": torch.get_num_threads(), "fold": FOLD,
                "n_warmup": N_WARM, "n_measured": N_MEAS})
    if platform.system() == "Windows":
        import subprocess
        res["power_scheme"] = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                                             text=True).stdout.strip()
    for name, spec in B.REGISTRY.items():
        if (names and name not in names) or not B.available(name):
            continue
        try:
            score = spec.loader(FOLD)
        except NotImplementedError as e:
            print(f"[skip] {e}")
            continue
        res[f"{name}/decision"] = bench(score, rows)
        if spec.tree:
            model, prep = tree_parts(name)
            if model is not None:
                if hasattr(model, "get_booster"):
                    model.get_booster().set_param({"device": "cpu"})
                ex = shap.TreeExplainer(model)
                res[f"{name}/decision+treeshap"] = bench(lambda r: (score(r), ex.shap_values(prep(r))), rows)
        for k in (f"{name}/decision", f"{name}/decision+treeshap"):
            if k in res:
                print(f"  {k:<36} {res[k]['mean_ms']:8.2f} ms  (p95 {res[k]['p95_ms']:7.2f})", flush=True)
    out_f.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"-> {out_f}")


if __name__ == "__main__":
    main(sys.argv[1:])   # no argument: every available model; names: measure those and merge
