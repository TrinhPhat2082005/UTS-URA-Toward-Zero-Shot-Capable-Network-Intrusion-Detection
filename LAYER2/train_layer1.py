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

Kien truc encoder lay tu zeroshot_benchmark/04_training_v2/ (models/autoencoder.py,
models/svdd.py). Config XGBoost lay TRUC TIEP tu LAYER1/03_training/common/xgb_config_mono.py
(ban Layer 1 cuoi cung): cung tham so cay + rang buoc DON DIEU TANG tren feature
augmented co ngu nghia "cang lon cang bat thuong" (AE: ae_recon_error; SVDD: K
sphere dist + d_min). Bo cuc rang buoc giong het LAYER1/03_training/train_{ae,svdd}_xgboost_mono.py.
Khac biet voi LAYER1: train mot lan tren tap full-label thay vi 14 fold, ep chay GPU,
va KHONG early stopping (co dinh 500 cay) vi val_full la tap danh gia RAG/VLM
(tren 42 checkpoint LOAO, early stopping cung khong bao gio cat qua 5 cay).

Encoder (AE/SVDD) mac dinh DUNG LAI ban da train trong thu muc artifact (rang buoc
don dieu chi doi XGBoost). --retrain-encoders de train lai tu dau.

MOT SCALER DUY NHAT
-------------------
RobustScaler fit tren benign-only, roi dung CHUNG cho ca encoder input lan
XGBoost input. Day la loi kien truc da tung mac o ban v1 (2 scaler khac nhau
cho cung mot tap) — xem docstring train_ae_xgboost.py. Giu dung mot scaler.

THRESHOLD
---------
Theo NGAN SACH FPR tren benign cua TRAIN: tau = Q_{1-alpha}(p(x), x benign train),
alpha = OPERATING_ALPHA (0.4%). Giong quy trinh danh gia Layer 1 cua paper (§4.4).
Val khong tham gia chon nguong (val la tap bao cao TP/FP cho Layer 2). metrics.json
luu them nguong cho moi alpha trong ALPHAS va nguong F1-max tren train de tham chieu;
key "threshold" (pipeline doc key nay) = nguong tai OPERATING_ALPHA.
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


def _load_layer1_config():
    """Nap common/xgb_config_mono.py cua Layer 1 (cung config da train 14 fold
    LOAO) theo duong dan file.

    Khong import kieu `common.xxx` vi zeroshot_benchmark/04_training_v2 cung co
    package ten `common` (da nam tren sys.path) -> trung ten.
    Tim theo thu tu: UTS_LAYER1_CODE, repo (LAYER1/03_training, chay tu repo hoac tu
    workspace), roi ban sao trong zeroshot_benchmark/04_training_v2 (giu ten cu
    xgb_config_v5_mono.py).
    """
    import importlib.util
    import os
    roots = [os.environ.get("UTS_LAYER1_CODE"), WS / "LAYER1" / "03_training",
             WS / "UTS_URA" / "LAYER1" / "03_training", WS / "zeroshot_benchmark" / "04_training_v2"]
    path = next(Path(r) / "common" / n for r in roots if r
                for n in ("xgb_config_mono.py", "xgb_config_v5_mono.py") if (Path(r) / "common" / n).exists())
    spec = importlib.util.spec_from_file_location("layer1_xgb_config_mono", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_L1CFG = _load_layer1_config()
XGB_PARAMS = _L1CFG.XGB_PARAMS
XGB_SCALE_POS_DIVISOR = _L1CFG.XGB_SCALE_POS_DIVISOR
build_monotone_constraints = _L1CFG.build_monotone_constraints

# Nguong theo ngan sach FPR tren benign cua train
ALPHAS = [0.001, 0.002, 0.004, 0.008, 0.010, 0.015, 0.020, 0.030]
OPERATING_ALPHA = 0.004

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
def _val_metrics(yva, p_va, thr):
    yh = (p_va >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yva, yh, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(yva, yh, zero_division=0)),
        "precision": float(precision_score(yva, yh, zero_division=0)),
        "recall": float(recall_score(yva, yh, zero_division=0)),
        "fpr": float(fp / (fp + tn)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train_xgb(Xtr, ytr, Xva, yva, names, tag, out_dir, mono):
    print(f"\n{'='*66}\n  XGBoost [{tag}] — {Xtr.shape[1]} feature\n{'='*66}")
    assert mono is not None and len(mono) == Xtr.shape[1], (
        f"monotone_constraints dai {None if mono is None else len(mono)} != {Xtr.shape[1]} cot")
    print(f"  monotone +1 tren {sum(mono)} cot: {[n for n, c in zip(names, mono) if c]}")
    pos = int(ytr.sum())
    spw = ((len(ytr) - pos) / pos) / XGB_SCALE_POS_DIVISOR
    print(f"  scale_pos_weight = {spw:.3f}  (chia {XGB_SCALE_POS_DIVISOR})")

    params = dict(XGB_PARAMS)
    params.pop("early_stopping_rounds", None)     # co dinh so cay, val_full chi de log
    params.update(device="cuda", tree_method="hist")

    t0 = time.time()
    clf = xgb.XGBClassifier(**params, scale_pos_weight=spw, monotone_constraints=mono)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=100)
    dt = time.time() - t0
    n_trees = clf.get_booster().num_boosted_rounds()
    print(f"  train {dt:.0f}s, {n_trees} cay")

    # Nguong theo ngan sach FPR tren benign cua TRAIN (val khong tham gia)
    p_tr = clf.predict_proba(Xtr)[:, 1]
    p_ben = p_tr[ytr == 0]
    thr_by_alpha = {f"{a:.3f}": float(np.quantile(p_ben, 1 - a, method="higher")) for a in ALPHAS}
    thr = thr_by_alpha[f"{OPERATING_ALPHA:.3f}"]
    thr_f1, f1_tr = best_f1_threshold(ytr, p_tr)
    print(f"  threshold (FPR {OPERATING_ALPHA:.1%} tren benign TRAIN) = {thr:.4f}   "
          f"[tham chieu F1-max tren TRAIN = {thr_f1:.4f}]")

    p_va = clf.predict_proba(Xva)[:, 1]
    val_by_alpha = {k: _val_metrics(yva, p_va, t) for k, t in thr_by_alpha.items()}
    m = {
        "n_features": int(Xtr.shape[1]),
        "threshold": thr,
        "threshold_rule": "fpr_budget_on_train_benign",
        "operating_alpha": OPERATING_ALPHA,
        "thresholds_by_alpha": thr_by_alpha,
        "threshold_f1max_train": thr_f1,
        "n_trees": int(n_trees),
        "early_stopping": False,
        "monotone_constraints": list(mono),
        "xgb_config": "LAYER1/03_training/common/xgb_config_mono.py",
        "train_seconds": round(dt, 1),
        "scale_pos_weight": round(spw, 4),
        "val": {**val_by_alpha[f"{OPERATING_ALPHA:.3f}"],
                "auroc": float(roc_auc_score(yva, p_va)),
                "auprc": float(average_precision_score(yva, p_va))},
        "val_by_alpha": val_by_alpha,
    }
    print(f"  {'alpha':>6} {'thr':>7} {'F1':>7} {'P':>7} {'R':>7} {'FPR':>8} {'TP':>7} {'FP':>6}")
    for k, v in val_by_alpha.items():
        print(f"  {float(k):>6.1%} {thr_by_alpha[k]:>7.4f} {v['f1']:>7.4f} {v['precision']:>7.4f} "
              f"{v['recall']:>7.4f} {v['fpr']:>8.5f} {v['tp']:>7,} {v['fp']:>6,}")
    print(f"  VAL AUROC={m['val']['auroc']:.4f}  AUPRC={m['val']['auprc']:.4f}")

    joblib.dump(clf, out_dir / "model.pkl")
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(names, f, indent=1)
    return m


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ae", "svdd"], default=None)
    ap.add_argument("--retrain-encoders", action="store_true",
                    help="train lai AE/SVDD thay vi dung ban da co trong thu muc artifact")
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
        reuse = (not args.retrain_encoders and (d / "ae_model.pt").exists()
                 and (d / "scaler.pkl").exists())
        if reuse:
            print(f"\n  Dung lai scaler + AE da train trong {d}")
            sc = joblib.load(d / "scaler.pkl")
        else:
            sc = fit_scaler(Xben_raw, d / "scaler.pkl")
        Xben_s, Xtr_s, Xva_s = scale(sc, Xben_raw), scale(sc, Xtr_raw), scale(sc, Xva_raw)

        if reuse:
            ae = BenignAutoencoder(input_dim=len(feats), bottleneck_dim=AE_BOTTLENECK).to(DEVICE)
            ae.load_state_dict(torch.load(d / "ae_model.pt", map_location=DEVICE))
            ae.eval()
            with open(d / "ae_history.json", encoding="utf-8") as f:
                best = json.load(f)["best_val_loss"]
        else:
            ae, hist, best = train_ae(Xben_s, d)
            with open(d / "ae_history.json", "w", encoding="utf-8") as f:
                json.dump({"best_val_loss": best, "history": hist}, f, indent=1)

        print("  sinh feature AE ...")
        Atr, Ava = ae_features(ae, Xtr_s), ae_features(ae, Xva_s)
        names = feats + [f"ae_latent_{i}" for i in range(AE_BOTTLENECK)] + ["ae_recon_error"]
        # [78 goc | 16 latent (tu do) | 1 recon_error (+1)] — nhu LAYER1/train_ae_xgboost.py
        mono = build_monotone_constraints(n_base=len(feats), n_augmented_monotone=1,
                                          n_augmented_free=AE_BOTTLENECK)
        m = train_xgb(np.hstack([Xtr_s, Atr]), ytr,
                      np.hstack([Xva_s, Ava]), yva, names, "AE-augmented", d, mono)
        m["encoder_reused"] = reuse
        m["ae_best_val_loss"] = best
        m["ae_bottleneck"] = AE_BOTTLENECK
        summary["variants"]["ae_xgboost"] = m
        with open(d / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)

    # ---------------- SVDD-augmented ----------------
    if want_svdd:
        d = OUT / "svdd_xgboost"
        d.mkdir(parents=True, exist_ok=True)
        old_m = {}
        if (d / "metrics.json").exists():
            with open(d / "metrics.json", encoding="utf-8") as f:
                old_m = json.load(f)
        reuse = (not args.retrain_encoders and (d / "svdd").is_dir()
                 and (d / "scaler.pkl").exists() and "svdd_K_total" in old_m)
        if reuse:
            print(f"\n  Dung lai scaler + Mixture SVDD da train trong {d}")
            sc = joblib.load(d / "scaler.pkl")
        else:
            sc = fit_scaler(Xben_raw, d / "scaler.pkl")
        Xben_s, Xtr_s, Xva_s = scale(sc, Xben_raw), scale(sc, Xtr_raw), scale(sc, Xva_raw)

        if reuse:
            K, k_chosen = int(old_m["svdd_K_total"]), int(old_m["svdd_k_chosen"])
            svdd = MixtureSVDD(K=K, input_dim=len(feats), embed_dim=SVDD_EMBED, device=str(DEVICE))
            svdd.load(str(d / "svdd"))
        else:
            svdd, K, k_chosen = train_svdd(Xben_s, d)

        print("  sinh feature SVDD ...")
        Str, Sva = svdd_features(svdd, Xtr_s), svdd_features(svdd, Xva_s)
        names = feats + [f"svdd_sphere_{i}" for i in range(K)] + ["svdd_d_min"]
        # [78 goc | K sphere dist (+1) | d_min (+1)] — nhu LAYER1/train_svdd_xgboost.py
        mono = build_monotone_constraints(n_base=len(feats), n_augmented_monotone=K + 1)
        m = train_xgb(np.hstack([Xtr_s, Str]), ytr,
                      np.hstack([Xva_s, Sva]), yva, names, "SVDD-augmented", d, mono)
        m["encoder_reused"] = reuse
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
