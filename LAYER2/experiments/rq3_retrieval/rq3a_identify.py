"""
RQ3a — nhan dien loai tan cong da biet (paper: Section 5.3.1, Table tab:res-ident).

Protocol B: query = moi alert TP tren val_full o nguong van hanh cua Layer 1; tap tham
chieu R = shap_pool_<variant>.npz (moi flow attack cua train_full, 122,090 vector).
Ba luat tren cung tap alert, cho hai khoa (vector SHAP / 78 feature da scale):

  l2      luat cua pipeline (Eq. type-dist, shap_knn_retriever.py): d_l = min L2 toi R_l,
          top-3 = 3 loai co d_l nho nhat                       -> Top-1, Hit@3, Technique
  cosine  baseline: S_l = max cosine toi R_l                    -> Top-1, Technique
  vote    baseline: da so phieu trong 10 lang gieng L2           -> Top-1, Technique,
          ti le alert ma 10 lang gieng cung mot nhan

Macro tren 12 loai (bo FTP-BruteForce va DoS-SlowHTTPTest [Liu2022]); in them macro 14.

Chay 2 lan (nap Layer 1 va tinh khoang cach trong cung tien trinh de het RAM may 16 GB):
    python LAYER2/experiments/rq3_retrieval/rq3a_identify.py ae_xgboost   # lan 1: SHAP + feature -> cache
    python LAYER2/experiments/rq3_retrieval/rq3a_identify.py ae_xgboost   # lan 2: cham -> json
Output: output/rq3a_identify/rq3a_<variant>.json
"""
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

paths.use_layer2_code()
import config  # noqa: E402
from shap_knn_retriever import LABEL_TO_PROFILE_ID  # noqa: E402

K = 10
OUT = paths.ensure(paths.OUT_RQ3A)
TECH = {l: json.loads((paths.PROFILES / f"{p}.json").read_text(encoding="utf-8"))["mitre_id"]
        for l, p in LABEL_TO_PROFILE_ID.items()}


def attack_rows(path, f):
    parts = []
    for ch in pd.read_csv(path, usecols=f + ["Label", "Label_Text"], chunksize=100_000):
        ch = ch[ch["Label"] == 1]
        parts.append((ch[f].to_numpy("float32"), ch["Label_Text"].to_numpy()))
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def build_cache(v, cache):
    """SHAP + feature da scale cua alert TP va cua R. Khong cham gi o buoc nay."""
    import pipeline
    l1 = pipeline.Layer1Engine(v)
    cols = list(pd.read_csv(config.VAL_BINARY_CSV, nrows=0).columns)
    f = l1.base_feature_columns(pd.DataFrame(columns=cols))
    X, y = attack_rows(config.VAL_BINARY_CSV, f)
    p = np.concatenate([l1.predict_proba(X[i:i + 5000]) for i in range(0, len(X), 5000)])
    keep = p >= config.layer1_threshold(v)
    X, y = X[keep], y[keep]
    sv = np.vstack([np.asarray(l1.score(X[i:i + 2000])[1], dtype="float32") for i in range(0, len(X), 2000)])
    Xr = np.vstack([l1.preprocess(X[i:i + 20000]).astype("float32") for i in range(0, len(X), 20000)])
    X = None
    gc.collect()

    npz = np.load(Path(config.ARTIFACTS_DIR) / f"shap_pool_{v}.npz", allow_pickle=True)
    keys = [k for k in npz.files if k.startswith("label__")]
    labs = np.array([k[len("label__"):] for k in keys])
    sizes = np.array([len(npz[k]) for k in keys])
    P = np.vstack([npz[k].astype("float32") for k in keys])
    # Cung cac flow attack cua train_full da dung build pool SHAP, cung thu tu nhan
    Xt, yt = attack_rows(config.TRAIN_FULL_CSV, f)
    Praw = np.vstack([l1.preprocess(Xt[yt == l]).astype("float32") for l in labs])
    assert len(Praw) == len(P), "pool feature tho khong khop pool SHAP"
    np.savez(cache, sv=sv, Xr=Xr, y=y, P=P, Praw=Praw, labs=labs, sizes=sizes)


def identify(P, yP, starts, labs, Q, chunk=8):
    """Ba luat tren cung mot lan tinh. Tra (top-3 L2, top-1 cosine, top-1 vote, so nhan trong 10 NN)."""
    sqP = (P ** 2).sum(1)
    normP = np.sqrt(sqP) + 1e-12
    t3, cos1, vote, ndist = [], [], [], []
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        dot = q @ P.T
        d = sqP[None, :] - 2.0 * dot + (q ** 2).sum(1)[:, None]
        t3.extend(labs[np.argsort(np.minimum.reduceat(d, starts, axis=1), 1)[:, :3]])
        sim = dot / (np.linalg.norm(q, axis=1)[:, None] + 1e-12) / normP[None, :]
        cos1.extend(labs[np.argmax(np.maximum.reduceat(sim, starts, axis=1), 1)])
        nn = np.argpartition(d, K, axis=1)[:, :K]
        nn = np.take_along_axis(nn, np.argsort(np.take_along_axis(d, nn, 1), 1), 1)
        for row in nn:
            l = yP[row]
            c = Counter(l)
            first = {}
            for r, x in enumerate(l):
                first.setdefault(x, r)
            vote.append(sorted(c, key=lambda x: (-c[x], first[x]))[0])
            ndist.append(len(c))
    return np.array(t3), np.array(cos1), np.array(vote), np.array(ndist)


def summarize(y, t3, cos1, vote):
    tech = lambda pred: [TECH[p] == TECH[a] for p, a in zip(pred, y)]
    df = pd.DataFrame({"y": y,
                       "l2_top1": t3[:, 0] == y, "l2_hit3": (t3 == y[:, None]).any(1), "l2_tech": tech(t3[:, 0]),
                       "cos_top1": cos1 == y, "cos_tech": tech(cos1),
                       "vote_top1": vote == y, "vote_tech": tech(vote)})
    per = df.groupby("y").mean()
    m12 = per.loc[[t for t in per.index if t not in paths.TWINS]].mean()
    return per, m12, per.mean()


def main():
    v = sys.argv[1]
    cache = OUT / f"cache_{v}.npz"
    if not cache.exists():
        build_cache(v, cache)
        print(f"cache -> {cache.name}; chay lai lenh nay de cham")
        return

    c = np.load(cache, allow_pickle=True)
    labs, sizes = c["labs"], c["sizes"]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    yP, y = np.repeat(labs, sizes), c["y"]
    out = {"variant": v, "n_alerts": int(len(y)), "n_pool": int(len(yP)),
           "n_per_type": {t: int((y == t).sum()) for t in np.unique(y)}}
    print(f"==== {v} | {len(y):,} alert TP | R = {len(yP):,} vector ====")
    for key, P, Q in [("shap", c["P"], c["sv"]), ("raw", c["Praw"], c["Xr"])]:
        t3, cos1, vote, ndist = identify(P, yP, starts, labs, Q)
        per, m12, m14 = summarize(y, t3, cos1, vote)
        out[key] = {"macro12": m12.to_dict(), "macro14": m14.to_dict(), "per_type": per.to_dict(),
                    "vote_single_label_share": float((ndist == 1).mean())}
        print(f"  [{key}] L2     top1 {m12.l2_top1:.4f}  hit3 {m12.l2_hit3:.4f}  tech {m12.l2_tech:.4f}"
              f"   | cosine top1 {m12.cos_top1:.4f} | vote top1 {m12.vote_top1:.4f}"
              f"   | 10 NN cung nhan: {(ndist == 1).mean():.3f}")
    (OUT / f"rq3a_{v}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {OUT / f'rq3a_{v}.json'}")


if __name__ == "__main__":
    main()
