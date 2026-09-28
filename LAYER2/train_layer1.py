"""
Train Layer 1 cho thi nghiem Layer 2 — 2 bien the XGBoost tang cuong.

CHAY:  python train_layer1.py            # ca AE lan SVDD
       python train_layer1.py --only ae
       python train_layer1.py --only svdd

CHUOI 4 BUOC
------------
  1. Loc benign-only tu train_full.csv (Label == 0), bo het dong attack.
  2. Train AE va SVDD tren CHI benign do (GPU).
  3. Dung AE/SVDD sinh feature tang cuong.
  4. Train XGBoost tren [78 feature goc | feature tang cuong] (GPU).

Kien truc + hyperparameter lay nguyen tu
zeroshot_benchmark/04_training_v2/ (models/autoencoder.py, models/svdd.py,
train_ae_only.py, train_svdd_only.py, common/xgb_config_v4_new_data.py).
Khac biet duy nhat: train mot lan tren tap full-label thay vi 14 fold, va ep
chay GPU.

MOT SCALER DUY NHAT
-------------------
RobustScaler fit tren benign-only, roi dung CHUNG cho ca encoder input lan
XGBoost input. Day la loi kien truc da tung mac o ban v1 (2 scaler khac nhau
cho cung mot tap) — xem docstring train_ae_xgboost.py. Giu dung mot scaler.

THRESHOLD
---------
Tune F1-max tren TRAIN, khong phai val. Val la tap dung de bao cao TP/FP cho
Layer 2; tune nguong tren chinh no roi bao cao lai tren no la leak.
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
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
WS = HERE.parent
BENCH = WS / "zeroshot_benchmark" / "04_training_v2"

sys.path.insert(0, str(BENCH))
from models.autoencoder import BenignAutoencoder            # noqa: E402
from models.svdd import MixtureSVDD                          # noqa: E402
from common.xgb_config_v4_new_data import (                  # noqa: E402
    XGB_PARAMS, XGB_SCALE_POS_DIVISOR)

# --------------------------------------------------------------------------
TRAIN_CSV = HERE / "data" / "train_full.csv"
VAL_CSV = HERE / "data" / "val_full.csv"
OUT = HERE / "artifacts" / "layer1"

SEED = 42
CLIP_VAL = 20.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AE
AE_BOTTLENECK = 16
AE_EPOCHS = 100
AE_BATCH = 2048
AE_LR = 1e-2
AE_PATIENCE = 7
AE_GRAD_CLIP = 1.0
AE_VAL_FRAC = 0.1

# SVDD
SVDD_EMBED = 64
SVDD_NU = 0.01
SVDD_LR = 1e-3
SVDD_EPOCHS = 50
SVDD_BATCH = 2048
SVDD_K_RANGE = range(2, 11)
SVDD_MIN_K = 6
SVDD_K_SPLIT = 3
SVDD_ELBOW_N = 50_000

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================================================
def feature_cols(df):
    return [c for c in df.columns if c not in ("Label", "Label_Text")]


def load_data():
    print(f"Doc {TRAIN_CSV.name} / {VAL_CSV.name} ...")
    tr = pd.read_csv(TRAIN_CSV)
    va = pd.read_csv(VAL_CSV)
    feats = feature_cols(tr)
    assert feature_cols(va) == feats, "cot train/val lech nhau"

    # Buoc 1: benign-only cho encoder
    ben = tr[tr["Label"] == 0]
    print(f"  train {len(tr):,} ({tr['Label'].sum():,} attack) | "
          f"val {len(va):,} ({va['Label'].sum():,} attack)")
    print(f"  benign-only cho AE/SVDD: {len(ben):,} dong, {len(feats)} feature")
    return tr, va, ben, feats


def clean(X):
    return np.nan_to_num(np.asarray(X, dtype="float64"),
                         nan=0.0, posinf=0.0, neginf=0.0)


def fit_scaler(X_benign, path):
    sc = RobustScaler().fit(X_benign)
    joblib.dump(sc, path)
    print(f"  RobustScaler fit tren benign-only -> {path.name}")
    return sc


def scale(sc, X):
    return np.clip(sc.transform(clean(X)), -CLIP_VAL, CLIP_VAL)


def best_f1_threshold(y, proba, n=200):
    """Quet nguong, tra (threshold, f1) tot nhat."""
    lo, hi = float(proba.min()), float(proba.max())
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(lo, hi, n):
        f = f1_score(y, (proba >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), float(f)
    return best_t, best_f


# ==========================================================================
# BUOC 2a: AE
# ==========================================================================
def train_ae(X_ben_s, out_dir):
    print(f"\n{'='*66}\n  AE tren benign-only ({len(X_ben_s):,} dong, {DEVICE})\n{'='*66}")
    tr, va = train_test_split(X_ben_s, test_size=AE_VAL_FRAC, random_state=SEED)

    dl_tr = DataLoader(TensorDataset(torch.tensor(tr, dtype=torch.float32)),
                       batch_size=AE_BATCH, shuffle=True, drop_last=True)
    dl_va = DataLoader(TensorDataset(torch.tensor(va, dtype=torch.float32)),
                       batch_size=AE_BATCH, shuffle=False)

    model = BenignAutoencoder(input_dim=X_ben_s.shape[1],
                              bottleneck_dim=AE_BOTTLENECK).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=AE_LR, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    crit = nn.MSELoss()

    best, bad, hist = float("inf"), 0, []
    t0 = time.time()
    for ep in range(1, AE_EPOCHS + 1):
        model.train()
        s = 0.0
        for (xb,) in dl_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            xh, _ = model(xb)
            loss = crit(xh, xb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), AE_GRAD_CLIP)
            opt.step()
            s += loss.item()
        tr_loss = s / len(dl_tr)

        model.eval()
        s = 0.0
        with torch.no_grad():
            for (xb,) in dl_va:
                xb = xb.to(DEVICE, non_blocking=True)
                xh, _ = model(xb)
                s += crit(xh, xb).item()
        va_loss = s / len(dl_va)
        sched.step(va_loss)
        hist.append({"epoch": ep, "train": tr_loss, "val": va_loss})

        if va_loss < best - 1e-6:
            best, bad = va_loss, 0
            torch.save(model.state_dict(), out_dir / "ae_model.pt")
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d}/{AE_EPOCHS}  train={tr_loss:.6f}  val={va_loss:.6f}")
        if bad >= AE_PATIENCE:
            print(f"  early stop @ epoch {ep} (val tot nhat {best:.6f})")
            break

    model.load_state_dict(torch.load(out_dir / "ae_model.pt", map_location=DEVICE))
    model.eval()
    print(f"  xong trong {time.time()-t0:.0f}s, val loss {best:.6f}")
    return model, hist, best


def ae_features(model, X_s, batch=8192):
    """Tra [latent_z (16) | recon_error (1)]."""
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_s), batch):
            xb = torch.tensor(X_s[i:i+batch], dtype=torch.float32).to(DEVICE)
            xh, z = model(xb)
            err = ((xb - xh) ** 2).mean(dim=1, keepdim=True)
            outs.append(torch.cat([z, err], dim=1).cpu().numpy())
    return np.vstack(outs)


# ==========================================================================
# BUOC 2b: SVDD
# ==========================================================================
def train_svdd(X_ben_s, out_dir):
    print(f"\n{'='*66}\n  SVDD tren benign-only ({len(X_ben_s):,} dong, {DEVICE})\n{'='*66}")
    from sklearn.metrics import silhouette_score

    idx = np.random.choice(len(X_ben_s), min(SVDD_ELBOW_N, len(X_ben_s)), replace=False)
    X_el = X_ben_s[idx]
    print(f"  chon K qua silhouette (K>{SVDD_MIN_K}) tren {len(X_el):,} mau...")
    sils = []
    for k in SVDD_K_RANGE:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=5, max_iter=200)
        lab = km.fit_predict(X_el)
        si = np.random.choice(len(X_el), min(10_000, len(X_el)), replace=False)
        s = float(silhouette_score(X_el[si], lab[si]))
        sils.append(s)
        print(f"    K={k:2d}  silhouette={s:.4f}")
    ks = list(SVDD_K_RANGE)
    ok = [i for i, k in enumerate(ks) if k > SVDD_MIN_K]
    k_chosen = ks[max(ok, key=lambda i: sils[i])]
    print(f"  K_CHOSEN = {k_chosen}")

    km_assign = KMeans(n_clusters=k_chosen, random_state=SEED, n_init=10)
    assign = km_assign.fit_predict(X_ben_s)
    raw = {k: X_ben_s[assign == k] for k in range(k_chosen)}
    for k, v in raw.items():
        print(f"    cluster {k}: {len(v):,}")

    big = max(raw, key=lambda k: len(raw[k]))
    km_sub = KMeans(n_clusters=SVDD_K_SPLIT, random_state=SEED, n_init=10)
    sub_a = km_sub.fit_predict(raw[big])
    print(f"  tach cluster lon nhat #{big} thanh {SVDD_K_SPLIT} sub-cluster")

    clusters = [raw[k] for k in range(k_chosen) if k != big]
    clusters += [raw[big][sub_a == j] for j in range(SVDD_K_SPLIT)]
    K_total = len(clusters)
    print(f"  tong {K_total} sphere")

    joblib.dump(km_assign, out_dir / "kmeans_assign.pkl")
    joblib.dump(km_sub, out_dir / "kmeans_sub.pkl")

    t0 = time.time()
    svdd = MixtureSVDD(K=K_total, input_dim=X_ben_s.shape[1], embed_dim=SVDD_EMBED,
                       nu=SVDD_NU, lr=SVDD_LR, epochs=SVDD_EPOCHS,
                       batch_size=SVDD_BATCH, device=str(DEVICE))
    svdd.fit(clusters)
    svdd.save(str(out_dir / "svdd"))
    print(f"  xong trong {time.time()-t0:.0f}s")
    return svdd, K_total, k_chosen


def svdd_features(svdd, X_s):
    """Tra [sphere_dist (K) | d_min (1)]."""
    d = svdd.all_sphere_distances(X_s)
    return np.hstack([d, d.min(axis=1, keepdims=True)])


# ==========================================================================
# BUOC 4: XGBoost tang cuong
# ==========================================================================
def train_xgb(Xtr, ytr, Xva, yva, names, tag, out_dir):
    print(f"\n{'='*66}\n  XGBoost [{tag}] — {Xtr.shape[1]} feature\n{'='*66}")
    pos = int(ytr.sum())
    spw = ((len(ytr) - pos) / pos) / XGB_SCALE_POS_DIVISOR
    print(f"  scale_pos_weight = {spw:.3f}  (chia {XGB_SCALE_POS_DIVISOR})")

    params = dict(XGB_PARAMS)
    params.update(device="cuda", tree_method="hist")

    t0 = time.time()
    clf = xgb.XGBClassifier(**params, scale_pos_weight=spw)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=50)
    dt = time.time() - t0
    print(f"  train {dt:.0f}s, {clf.best_iteration + 1} cay")

    # Threshold tune tren TRAIN (khong phai val — val la tap bao cao TP/FP)
    p_tr = clf.predict_proba(Xtr)[:, 1]
    thr, f1_tr = best_f1_threshold(ytr, p_tr)
    print(f"  threshold (F1-max tren TRAIN) = {thr:.4f}  (F1_train={f1_tr:.4f})")

    p_va = clf.predict_proba(Xva)[:, 1]
    yh = (p_va >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yva, yh).ravel()
    m = {
        "n_features": int(Xtr.shape[1]),
        "threshold": thr,
        "best_iteration": int(clf.best_iteration),
        "train_seconds": round(dt, 1),
        "scale_pos_weight": round(spw, 4),
        "val": {
            "f1": float(f1_score(yva, yh)),
            "precision": float(precision_score(yva, yh, zero_division=0)),
            "recall": float(recall_score(yva, yh)),
            "fpr": float(fp / (fp + tn)),
            "auroc": float(roc_auc_score(yva, p_va)),
            "auprc": float(average_precision_score(yva, p_va)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
    }
    print(f"  VAL  F1={m['val']['f1']:.4f}  P={m['val']['precision']:.4f}  "
          f"R={m['val']['recall']:.4f}  FPR={m['val']['fpr']:.5f}  "
          f"AUROC={m['val']['auroc']:.4f}")
    print(f"       TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

    joblib.dump(clf, out_dir / "model.pkl")
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(names, f, indent=1)
    return m


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ae", "svdd"], default=None)
    args = ap.parse_args()
    want_ae = args.only in (None, "ae")
    want_svdd = args.only in (None, "svdd")

    print(f"DEVICE = {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU    = {torch.cuda.get_device_name(0)}")
    OUT.mkdir(parents=True, exist_ok=True)

    tr, va, ben, feats = load_data()
    Xtr_raw, ytr = clean(tr[feats]), tr["Label"].to_numpy()
    Xva_raw, yva = clean(va[feats]), va["Label"].to_numpy()
    Xben_raw = clean(ben[feats])

    summary = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_train": int(len(tr)), "n_val": int(len(va)),
        "n_benign_for_encoder": int(len(ben)),
        "n_base_features": len(feats),
        "seed": SEED,
        "variants": {},
    }

    # ---------------- AE-augmented ----------------
    if want_ae:
        d = OUT / "ae_xgboost"
        d.mkdir(parents=True, exist_ok=True)
        sc = fit_scaler(Xben_raw, d / "scaler.pkl")
        Xben_s, Xtr_s, Xva_s = scale(sc, Xben_raw), scale(sc, Xtr_raw), scale(sc, Xva_raw)

        ae, hist, best = train_ae(Xben_s, d)
        with open(d / "ae_history.json", "w", encoding="utf-8") as f:
            json.dump({"best_val_loss": best, "history": hist}, f, indent=1)

        print("  sinh feature AE ...")
        Atr, Ava = ae_features(ae, Xtr_s), ae_features(ae, Xva_s)
        names = feats + [f"ae_latent_{i}" for i in range(AE_BOTTLENECK)] + ["ae_recon_error"]
        m = train_xgb(np.hstack([Xtr_s, Atr]), ytr,
                      np.hstack([Xva_s, Ava]), yva, names, "AE-augmented", d)
        m["ae_best_val_loss"] = best
        m["ae_bottleneck"] = AE_BOTTLENECK
        summary["variants"]["ae_xgboost"] = m
        with open(d / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)

    # ---------------- SVDD-augmented ----------------
    if want_svdd:
        d = OUT / "svdd_xgboost"
        d.mkdir(parents=True, exist_ok=True)
        sc = fit_scaler(Xben_raw, d / "scaler.pkl")
        Xben_s, Xtr_s, Xva_s = scale(sc, Xben_raw), scale(sc, Xtr_raw), scale(sc, Xva_raw)

        svdd, K, k_chosen = train_svdd(Xben_s, d)

        print("  sinh feature SVDD ...")
        Str, Sva = svdd_features(svdd, Xtr_s), svdd_features(svdd, Xva_s)
        names = feats + [f"svdd_sphere_{i}" for i in range(K)] + ["svdd_d_min"]
        m = train_xgb(np.hstack([Xtr_s, Str]), ytr,
                      np.hstack([Xva_s, Sva]), yva, names, "SVDD-augmented", d)
        m["svdd_K_total"] = K
        m["svdd_k_chosen"] = k_chosen
        summary["variants"]["svdd_xgboost"] = m
        with open(d / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)

    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*66}")
    for k, v in summary["variants"].items():
        print(f"  {k:<16} F1={v['val']['f1']:.4f}  FPR={v['val']['fpr']:.5f}  "
              f"FP={v['val']['fp']:,}  TP={v['val']['tp']:,}  thr={v['threshold']:.3f}")
    print(f"  artifacts -> {OUT}")


if __name__ == "__main__":
    main()
