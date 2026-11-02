"""Quet threshold tren checkpoint v2 (xgboost_only, ae_xgboost_mono, svdd_xgboost_mono).
Luu proba (val benign / val attack / test_zeroshot) moi fold ra npz."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402  (experiments/paths.py)
import sys
from pathlib import Path
import numpy as np, pandas as pd, joblib, torch

paths.use_layer1_code()
from models.autoencoder import BenignAutoencoder
from models.svdd import MixtureSVDD

SPL = paths.SPLITS
ART = paths.L1_ART
OUT = paths.ensure(paths.OUT_L1) / "sweep_v2_scores.npz"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = sorted(p.name for p in SPL.iterdir() if p.is_dir())
CLIP = 20.0

def load(label, name):
    df = pd.read_csv(SPL / label / name)
    feats = [c for c in df.columns if c not in ("Label", "Label_Text")]
    return df[feats].values.astype("float32"), df["Label"].values, feats

ae_sc = joblib.load(ART / "ae_only/checkpoint/scaler.pkl")
sv_sc = joblib.load(ART / "svdd_only/checkpoint/scaler.pkl")
k_in = len(list((ART / "svdd_only/checkpoint/svdd_model").glob("encoder_*.pt")))

def ae_aug(ae, Xs):
    zs, es = [], []
    with torch.no_grad():
        for i in range(0, len(Xs), 8192):
            xb = torch.tensor(Xs[i:i+8192], device=DEV)
            xh, z = ae(xb)
            zs.append(z.cpu().numpy()); es.append(((xb - xh) ** 2).mean(1).cpu().numpy())
    return np.concatenate([Xs, np.concatenate(zs), np.concatenate(es)[:, None]], 1)

scores = {}
ae = svdd = None
for lab in LABELS:
    Xv, yv, feats = load(lab, "val_binary.csv")
    Xt, _, _ = load(lab, "test_zeroshot.csv")
    if ae is None:
        ae = BenignAutoencoder(input_dim=len(feats), bottleneck_dim=16).to(DEV)
        ae.load_state_dict(torch.load(ART / "ae_only/checkpoint/ae_best.pt", map_location=DEV))
        ae.eval()
        svdd = MixtureSVDD(K=k_in, input_dim=len(feats), embed_dim=64, device=DEV)
        svdd.load(str(ART / "svdd_only/checkpoint/svdd_model"))
    for var in ["xgboost_only", "ae_xgboost_mono", "svdd_xgboost_mono"]:
        ck = ART / var / lab / "checkpoint"
        m = joblib.load(ck / "model.pkl")
        out = []
        for X in (Xv, Xt):
            if var == "xgboost_only":
                A = joblib.load(ck / "scaler.pkl").transform(X)
            elif var == "ae_xgboost_mono":
                A = ae_aug(ae, np.clip(ae_sc.transform(X), -CLIP, CLIP).astype("float32"))
            else:
                Xs = np.clip(sv_sc.transform(X), -CLIP, CLIP)
                d = svdd.all_sphere_distances(Xs)
                A = np.concatenate([Xs, d, d.min(1, keepdims=True)], 1)
            out.append(m.predict_proba(A)[:, 1])
        scores[f"{var}|{lab}|vb"] = out[0][yv == 0]
        scores[f"{var}|{lab}|va"] = out[0][yv == 1]
        scores[f"{var}|{lab}|zs"] = out[1]
        print(var, lab, "zs@0.5=%.3f" % (out[1] >= 0.5).mean(), flush=True)
np.savez_compressed(OUT, **scores)
print("saved", OUT)
