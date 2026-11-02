"""Cham AE-only (reconstruction error) va SVDD-only (d_min) theo CUNG quy trinh voi 4 model kia.
Hai model nay train 1 lan tren benign -> diem benign-train dung chung 14 fold.
Doc sweep_train_scores.npz (4 model), them 2 model -> sweep_all_scores.npz. Khong sua gi trong project."""
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

SPL = paths.SPLITS_ROOT
ART = paths.L1_ART
HERE = paths.ensure(paths.OUT_L1)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = sorted(p.name for p in (SPL / "by_label").iterdir() if p.is_dir())
CLIP = 20.0

S = dict(np.load(HERE / "sweep_train_scores.npz"))

ae_sc = joblib.load(ART / "ae_only/checkpoint/scaler.pkl")
sv_sc = joblib.load(ART / "svdd_only/checkpoint/scaler.pkl")
k_in = len(list((ART / "svdd_only/checkpoint/svdd_model").glob("encoder_*.pt")))
ae = svdd = None


def ae_score(X):
    Xs = np.clip(ae_sc.transform(X), -CLIP, CLIP).astype("float32")
    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), 8192):
            xb = torch.tensor(Xs[i:i+8192], device=DEV)
            xh, _ = ae(xb)
            out.append(((xb - xh) ** 2).mean(1).cpu().numpy())
    return np.concatenate(out)


def svdd_score(X):
    return svdd.all_sphere_distances(np.clip(sv_sc.transform(X), -CLIP, CLIP)).min(1)


tb = pd.read_csv(SPL / "train_benign_only.csv")
feats = [c for c in tb.columns if c not in ("Label", "Label_Text")]
ae = BenignAutoencoder(input_dim=len(feats), bottleneck_dim=16).to(DEV)
ae.load_state_dict(torch.load(ART / "ae_only/checkpoint/ae_best.pt", map_location=DEV))
ae.eval()
svdd = MixtureSVDD(K=k_in, input_dim=len(feats), embed_dim=64, device=DEV)
svdd.load(str(ART / "svdd_only/checkpoint/svdd_model"))

Xtb = tb[feats].values.astype("float32")
del tb
tb_ae, tb_sv = ae_score(Xtb), svdd_score(Xtb)
del Xtb

for lab in LABELS:
    va = pd.read_csv(SPL / "by_label" / lab / "val_binary.csv")
    zs = pd.read_csv(SPL / "by_label" / lab / "test_zeroshot.csv")
    Xv, yv, Xz = va[feats].values.astype("float32"), va["Label"].values, zs[feats].values.astype("float32")
    for var, fn, tbs in (("ae_only", ae_score, tb_ae), ("svdd_only", svdd_score, tb_sv)):
        sv, sz = fn(Xv), fn(Xz)
        S[f"{var}|{lab}|tb"] = tbs
        S[f"{var}|{lab}|vb"], S[f"{var}|{lab}|va"], S[f"{var}|{lab}|zs"] = sv[yv == 0], sv[yv == 1], sz
    print("done", lab, flush=True)

np.savez_compressed(HERE / "sweep_all_scores.npz", **S)
print("saved")
