"""Tinh diem tren BENIGN CUA TRAIN (560K, dung chung 14 fold) cho 4 model,
va diem val/zs cho CLAD. Ket hop voi sweep_v2_scores.npz -> sweep_train_scores.npz.
Chi doc checkpoint + CSV, khong sua gi trong project."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402  (experiments/paths.py)
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, joblib, torch

paths.use_layer1_code()
from models.autoencoder import BenignAutoencoder
from models.svdd import MixtureSVDD
from models.clad.model import build_model
from models.clad.util.distance import cosine_scores_to_centroid

SPL = paths.SPLITS_ROOT
ART = paths.L1_ART
HERE = paths.ensure(paths.OUT_L1)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = sorted(p.name for p in (SPL / "by_label").iterdir() if p.is_dir())
CLIP = 20.0

scores = dict(np.load(HERE / "sweep_v2_scores.npz"))

tb = pd.read_csv(SPL / "train_benign_only.csv")
feats = [c for c in tb.columns if c not in ("Label", "Label_Text")]
assert (tb["Label"] == 0).all()
Xtb = tb[feats].values.astype("float32")
del tb
print("train benign rows:", len(Xtb), flush=True)

# Feature augmented cua benign train khong phu thuoc fold -> tinh 1 lan
ae_sc = joblib.load(ART / "ae_only/checkpoint/scaler.pkl")
ae = BenignAutoencoder(input_dim=len(feats), bottleneck_dim=16).to(DEV)
ae.load_state_dict(torch.load(ART / "ae_only/checkpoint/ae_best.pt", map_location=DEV))
ae.eval()
Xs = np.clip(ae_sc.transform(Xtb), -CLIP, CLIP).astype("float32")
zs, es = [], []
with torch.no_grad():
    for i in range(0, len(Xs), 8192):
        xb = torch.tensor(Xs[i:i+8192], device=DEV)
        xh, z = ae(xb)
        zs.append(z.cpu().numpy()); es.append(((xb - xh) ** 2).mean(1).cpu().numpy())
A_ae = np.concatenate([Xs, np.concatenate(zs), np.concatenate(es)[:, None]], 1)

sv_sc = joblib.load(ART / "svdd_only/checkpoint/scaler.pkl")
k_in = len(list((ART / "svdd_only/checkpoint/svdd_model").glob("encoder_*.pt")))
svdd = MixtureSVDD(K=k_in, input_dim=len(feats), embed_dim=64, device=DEV)
svdd.load(str(ART / "svdd_only/checkpoint/svdd_model"))
Xs = np.clip(sv_sc.transform(Xtb), -CLIP, CLIP)
d = svdd.all_sphere_distances(Xs)
A_sv = np.concatenate([Xs, d, d.min(1, keepdims=True)], 1)
del Xs, d


def clad_scores(model, centroid, X):
    zl = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            zl.append(model(torch.tensor(X[i:i+8192], device=DEV)).cpu())
    return -cosine_scores_to_centroid(torch.cat(zl), centroid)   # cao = attack


for lab in LABELS:
    ck = lambda v: ART / v / lab / "checkpoint"
    # XGB-only: scaler rieng tung fold, khong clip
    m = joblib.load(ck("xgboost_only") / "model.pkl")
    scores[f"xgboost_only|{lab}|tb"] = m.predict_proba(joblib.load(ck("xgboost_only") / "scaler.pkl").transform(Xtb))[:, 1]
    scores[f"ae_xgboost_mono|{lab}|tb"] = joblib.load(ck("ae_xgboost_mono") / "model.pkl").predict_proba(A_ae)[:, 1]
    scores[f"svdd_xgboost_mono|{lab}|tb"] = joblib.load(ck("svdd_xgboost_mono") / "model.pkl").predict_proba(A_sv)[:, 1]

    # CLAD: scaler rieng tung fold, clip +-20
    sc = joblib.load(ck("clad") / "scaler.pkl")
    meta = json.loads((ck("clad") / "meta.json").read_text(encoding="utf-8"))
    cm = build_model(d_in=len(feats), hidden_dim=512, n_layers=4, d_out=32, dropout=0.0, block_norm=False).to(DEV)
    cm.load_state_dict(torch.load(ck("clad") / "clad_best.pt", map_location=DEV, weights_only=False))
    cm.eval()
    cen = torch.tensor(meta["centroid"], dtype=torch.float32)
    prep = lambda X: np.clip(sc.transform(X), -CLIP, CLIP).astype("float32")
    scores[f"clad|{lab}|tb"] = clad_scores(cm, cen, prep(Xtb))
    vdf = pd.read_csv(SPL / "by_label" / lab / "val_binary.csv")
    Xv, yv = vdf[feats].values.astype("float32"), vdf["Label"].values
    s = clad_scores(cm, cen, prep(Xv))
    scores[f"clad|{lab}|vb"], scores[f"clad|{lab}|va"] = s[yv == 0], s[yv == 1]
    Xt = pd.read_csv(SPL / "by_label" / lab / "test_zeroshot.csv")[feats].values.astype("float32")
    scores[f"clad|{lab}|zs"] = clad_scores(cm, cen, prep(Xt))
    print("done", lab, flush=True)

np.savez_compressed(HERE / "sweep_train_scores.npz", **scores)
print("saved")
