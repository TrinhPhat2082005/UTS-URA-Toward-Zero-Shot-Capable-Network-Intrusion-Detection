"""
RQ5 — latency cua duong phat hien (paper: Section 4.7, Table tab:res-latency, cot latency
cua tab:res-known va tab:res-ident).

Moi lan goi xu ly MOT flow/alert nhu khi trien khai; 20 lan khoi dong + 300 lan do,
time.perf_counter, CPU (GPU bi an). Bao cao mean, p95.

  Layer 1, Protocol B (pipeline.Layer1Engine)  : quyet dinh; quyet dinh + TreeSHAP chinh xac
  Layer 1 baseline (checkpoint fold Bot)       : XGBoost-only (+ TreeSHAP), AE-only, SVDD-only, CLAD
  Retrieval (SHAPKNNRetriever, R = 122,090)    : mot lan tinh L2 -> top-3 + novelty,
                                                 khoa SHAP va khoa feature tho, do xen ke

CHAY:  python LAYER2/experiments/rq5_latency/latency_detection_path.py
Output: output/rq5_latency/latency.json
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # do tren CPU nhu Section 4.1

import joblib
import numpy as np
import pandas as pd
import shap
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

paths.use_layer2_code()
paths.use_layer1_code()
import config    # noqa: E402
import pipeline  # noqa: E402
from retrieval import reference_set                   # noqa: E402
from retrieval.retriever import SHAPKNNRetriever      # noqa: E402
from models.autoencoder import BenignAutoencoder       # noqa: E402
from models.svdd import MixtureSVDD                    # noqa: E402
from models.clad.model import build_model              # noqa: E402
from models.clad.util.distance import cosine_scores_to_centroid  # noqa: E402

N_WARM, N_MEAS, FOLD, CLIP = 20, 300, "Bot", 20.0
ART = paths.L1_ART
OUT = paths.ensure(paths.OUT_RQ5)
torch.set_num_threads(4)


def bench(fn, items):
    ts = []
    for i, x in enumerate(items):
        t0 = time.perf_counter()
        fn(x)
        if i >= N_WARM:
            ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"mean_ms": float(ts.mean()), "p95_ms": float(np.percentile(ts, 95)), "n": int(len(ts))}


def main():
    va = pd.read_csv(config.VAL_BINARY_CSV, nrows=20_000)
    idx = np.random.default_rng(0).choice(len(va), N_WARM + N_MEAS, replace=False)
    res = {"cpu": platform.processor(), "torch_threads": torch.get_num_threads()}
    if platform.system() == "Windows":          # che do nguon anh huong latency CPU laptop
        import subprocess
        res["power_scheme"] = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                                             text=True).stdout.strip()

    # --- Layer 1, Protocol B ---
    for v in ["ae_xgboost", "svdd_xgboost"]:
        l1 = pipeline.Layer1Engine(v)
        l1.model.get_booster().set_param({"device": "cpu"})
        X = va[l1.base_feature_columns(va)].to_numpy("float64")[idx]
        rows = [X[i:i + 1] for i in range(len(X))]
        res[f"layer1/{v}/decision"] = bench(l1.predict_proba, rows)
        res[f"layer1/{v}/decision+treeshap"] = bench(l1.score, rows)

    # --- Layer 1 baselines (checkpoint fold Bot cua Protocol A) ---
    feats = [c for c in va.columns if c not in config.META_COLS]
    X = va[feats].to_numpy("float32")[idx]
    rows = [X[i:i + 1] for i in range(len(X))]
    m = joblib.load(ART / "xgboost_only" / FOLD / "checkpoint" / "model.pkl")
    m.get_booster().set_param({"device": "cpu"})
    sc = joblib.load(ART / "xgboost_only" / FOLD / "checkpoint" / "scaler.pkl")
    ex = shap.TreeExplainer(m)
    res["layer1/xgboost_only/decision"] = bench(lambda r: m.predict_proba(sc.transform(r)), rows)
    res["layer1/xgboost_only/decision+treeshap"] = bench(
        lambda r: (m.predict_proba(sc.transform(r)), ex.shap_values(sc.transform(r))), rows)

    ae_sc = joblib.load(ART / "ae_only/checkpoint/scaler.pkl")
    ae = BenignAutoencoder(len(feats), 16)
    ae.load_state_dict(torch.load(ART / "ae_only/checkpoint/ae_best.pt", map_location="cpu"))
    ae.eval()

    def ae_score(r):
        with torch.no_grad():
            x = torch.tensor(np.clip(ae_sc.transform(r), -CLIP, CLIP), dtype=torch.float32)
            xh, _ = ae(x)
            return ((x - xh) ** 2).mean()
    res["layer1/ae_only/decision"] = bench(ae_score, rows)

    sv_sc = joblib.load(ART / "svdd_only/checkpoint/scaler.pkl")
    k_in = len(list((ART / "svdd_only/checkpoint/svdd_model").glob("encoder_*.pt")))
    sv = MixtureSVDD(K=k_in, input_dim=len(feats), embed_dim=64, device="cpu")
    sv.load(str(ART / "svdd_only/checkpoint/svdd_model"))
    res["layer1/svdd_only/decision"] = bench(
        lambda r: sv.all_sphere_distances(np.clip(sv_sc.transform(r), -CLIP, CLIP)).min(1), rows)

    cd = ART / "clad" / FOLD / "checkpoint"
    meta = json.loads((cd / "meta.json").read_text())
    cm = build_model(d_in=len(feats), hidden_dim=512, n_layers=4, d_out=32, dropout=0.0, block_norm=False)
    cm.load_state_dict(torch.load(cd / "clad_best.pt", map_location="cpu", weights_only=False))
    cm.eval()
    csc = joblib.load(cd / "scaler.pkl")
    cen = torch.tensor(meta["centroid"], dtype=torch.float32)

    def clad_score(r):
        with torch.no_grad():
            z = cm(torch.tensor(np.clip(csc.transform(r), -CLIP, CLIP), dtype=torch.float32))
            return -cosine_scores_to_centroid(z, cen)
    res["layer1/clad/decision"] = bench(clad_score, rows)

    # --- Retrieval: khoa SHAP (pipeline) va khoa feature tho, do xen ke tren cung alert ---
    for v in ["ae_xgboost", "svdd_xgboost"]:
        l1 = pipeline.Layer1Engine(v)
        l1.model.get_booster().set_param({"device": "cpu"})
        f78 = l1.base_feature_columns(va)
        Xq = va[f78].to_numpy("float64")[idx]
        SQ = np.vstack([np.asarray(l1.score(Xq[i:i + 1])[1], dtype="float32") for i in range(len(Xq))])
        RQ = l1.preprocess(Xq).astype("float32")
        npz = reference_set.protocol_b_path(v)
        ret = SHAPKNNRetriever(str(npz), config.ATTACK_PROFILES_DIR, k=config.RAG_NOVELTY_K)
        tr = pd.read_csv(config.TRAIN_FULL_CSV, usecols=list(f78) + ["Label", "Label_Text"])
        tr = tr[tr["Label"] == 1]
        # Cung luat va cung thu tu nhan, R = feature da scale cua cung cac flow attack
        raw = SHAPKNNRetriever.__new__(SHAPKNNRetriever)
        raw.k, raw.labels, raw.sizes, raw.starts = ret.k, ret.labels, ret.sizes, ret.starts
        raw.pool = np.vstack([l1.preprocess(tr.loc[tr["Label_Text"] == l, f78].to_numpy("float64"))
                              .astype("float32") for l in ret.labels])
        raw.pool_sq = (raw.pool ** 2).sum(axis=1)
        tr = None
        t_s, t_r = [], []
        for i in range(len(SQ)):
            t0 = time.perf_counter()
            ret.score(SQ[i:i + 1])
            t1 = time.perf_counter()
            raw.score(RQ[i:i + 1])
            t2 = time.perf_counter()
            if i >= N_WARM:
                t_s.append((t1 - t0) * 1000)
                t_r.append((t2 - t1) * 1000)
        for key, ts in (("shap", t_s), ("raw", t_r)):
            ts = np.array(ts)
            res[f"retrieval/{v}/{key}"] = {"mean_ms": float(ts.mean()), "p95_ms": float(np.percentile(ts, 95)),
                                           "n": int(len(ts))}

    for k, r in res.items():
        if isinstance(r, dict):
            print(f"  {k:<42} {r['mean_ms']:7.2f} ms  (p95 {r['p95_ms']:6.2f})")
    (OUT / "latency.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"-> {OUT / 'latency.json'}")


if __name__ == "__main__":
    main()
