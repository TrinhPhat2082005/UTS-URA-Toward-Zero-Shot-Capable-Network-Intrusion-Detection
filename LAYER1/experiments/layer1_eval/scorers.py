"""Moi model Layer 1: checkpoint -> ham score(X), X = 78 feature CICFlowMeter tho, diem cao = attack.

Hai bo checkpoint, cung bo cuc thu muc:
    "paper"           paths.L1_ART    run cua paper (04_training_v2)
    "baseline-study"  paths.BASE_ART  run cua teammate (LAYER1/baseline_study)
        <dir>/<fold>/checkpoint/...   model train rieng moi fold
        <dir>/checkpoint/...          model one-class, mot checkpoint cho ca 14 fold
Tien xu ly giong luc train: moi model dung scaler cua chinh no; XGBoost-only, LightGBM, KNN, SVM,
Random Forest khong cat; cac model co encoder AE / SVDD / VAE / CLAD cat [-20, 20].

Ba model duoc dung lai tu thong tin luu trong checkpoint (code train khong kem theo):
  - KNN (PyTorchKNNClassifier, pickle duoi __main__): 50K flow train da scale, k = 5, vote theo 1/khoang cach.
  - VAE: kien truc suy tu ten/shape tham so trong vae_best.pt; diem = MSE tai tao + (0.1/78) * KL.
  - Deep SVDD (Ruff et al.): encoder khong bias 78-256-128-64, LeakyReLU 0.1, tam co dinh luu trong checkpoint.
extras.verify_reproduction() xac nhan KNN va VAE khop metrics.json cua nguoi train; deep_svdd_paper/ khong co
metrics.json nen Deep SVDD khong kiem duoc theo cach nay.
"""
import json
import sys
from dataclasses import dataclass
from typing import Callable

import joblib
import numpy as np

from . import paths

CLIP = 20.0
N_FEATURES = 78
ROOT = {"paper": paths.L1_ART, "baseline-study": paths.BASE_ART}


def _torch():
    """torch va thiet bi; CPU khi khong thay GPU (vd CUDA_VISIBLE_DEVICES="")."""
    import torch
    gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    return torch, torch.device("cuda" if gpu else "cpu")


def _proba(m, A):
    pos = int(np.flatnonzero(m.classes_ == 1)[0]) if hasattr(m, "classes_") else 1
    return m.predict_proba(A)[:, pos]


# ------------------------------------------------------------------ encoder dung chung (moi root mot ban)
_ENC = {}


def ae_encoder(root):
    """AE benign: X -> (X da scale, latent 16, loi tai tao)."""
    key = ("ae", str(root))
    if key not in _ENC:
        paths.use_layer1_code()
        from models.autoencoder import BenignAutoencoder
        torch, dev = _torch()
        d = root / "ae_only" / "checkpoint"
        sc = joblib.load(d / "scaler.pkl")
        ae = BenignAutoencoder(input_dim=N_FEATURES, bottleneck_dim=16).to(dev)
        ae.load_state_dict(torch.load(d / "ae_best.pt", map_location=dev))
        ae.eval()

        def forward(X):
            Xs = np.clip(sc.transform(X), -CLIP, CLIP).astype("float32")
            zs, es = [], []
            with torch.no_grad():
                for i in range(0, len(Xs), 8192):
                    xb = torch.tensor(Xs[i:i + 8192], device=dev)
                    xh, z = ae(xb)
                    zs.append(z.cpu().numpy())
                    es.append(((xb - xh) ** 2).mean(1).cpu().numpy())
            return Xs, np.concatenate(zs), np.concatenate(es)
        _ENC[key] = forward
    return _ENC[key]


def svdd_encoder(root):
    """Mixture Deep SVDD: X -> (X da scale, khoang cach toi tung sphere)."""
    key = ("svdd", str(root))
    if key not in _ENC:
        paths.use_layer1_code()
        from models.svdd import MixtureSVDD
        _, dev = _torch()
        d = root / "svdd_only" / "checkpoint"
        sc = joblib.load(d / "scaler.pkl")
        k_in = len(list((d / "svdd_model").glob("encoder_*.pt")))
        sv = MixtureSVDD(K=k_in, input_dim=N_FEATURES, embed_dim=64, device=str(dev))
        sv.load(str(d / "svdd_model"))

        def forward(X):
            Xs = np.clip(sc.transform(X), -CLIP, CLIP)
            return Xs, sv.all_sphere_distances(Xs)
        _ENC[key] = forward
    return _ENC[key]


def deep_svdd_encoder(root):
    """One-Class Deep SVDD (Ruff et al., ICML 2018): X -> binh phuong khoang cach toi tam."""
    key = ("deep_svdd", str(root))
    if key not in _ENC:
        import torch.nn as nn
        torch, dev = _torch()
        d = root / "deep_svdd_paper" / "checkpoint"
        sc = joblib.load(d / "scaler.pkl")
        ck = torch.load(d / "oneclass_deep_svdd.pt", map_location=dev, weights_only=False)
        net = nn.Sequential(
            nn.Linear(ck["input_dim"], 256, bias=False), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(256, 128, bias=False), nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(128, ck["embed_dim"], bias=False),
        ).to(dev)
        net.load_state_dict({k.replace("net.", "", 1): v for k, v in ck["encoder_state_dict"].items()})
        net.eval()
        centre = ck["center"].to(dev)

        def forward(X):
            Xs = np.clip(sc.transform(X), -CLIP, CLIP).astype("float32")
            out = []
            with torch.no_grad():
                for i in range(0, len(Xs), 8192):
                    z = net(torch.tensor(Xs[i:i + 8192], device=dev))
                    out.append(((z - centre) ** 2).sum(1).cpu().numpy())
            return np.concatenate(out)
        _ENC[key] = forward
    return _ENC[key]


def _vae_net(d_in=N_FEATURES, hidden=(512, 256), latent=32, slope=0.2, p=0.1):
    import torch.nn as nn

    def block(i, o):
        return [nn.Linear(i, o), nn.LayerNorm(o), nn.LeakyReLU(slope), nn.Dropout(p)]

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_backbone = nn.Sequential(*block(d_in, hidden[0]), *block(hidden[0], hidden[1]))
            self.fc_mu = nn.Linear(hidden[1], latent)
            self.fc_logvar = nn.Linear(hidden[1], latent)
            self.decoder = nn.Sequential(*block(latent, hidden[1]), *block(hidden[1], hidden[0]),
                                         nn.Linear(hidden[0], d_in))
    return VAE()


VAE_KL_WEIGHT = 0.1 / N_FEATURES     # diem VAE = (SSE + 0.1 * KL) / 78; dang nay moi khop metrics.json


def vae_encoder(root):
    """VAE benign: X -> (X da scale, mu, diem bat thuong)."""
    key = ("vae", str(root))
    if key not in _ENC:
        torch, dev = _torch()
        d = root / "vae_only" / "checkpoint"
        sc = joblib.load(d / "scaler.pkl")
        net = _vae_net().to(dev)
        net.load_state_dict(torch.load(d / "vae_best.pt", map_location=dev))
        net.eval()

        def forward(X):
            Xs = np.clip(sc.transform(X), -CLIP, CLIP).astype("float32")
            mus, scores = [], []
            with torch.no_grad():
                for i in range(0, len(Xs), 8192):
                    xb = torch.tensor(Xs[i:i + 8192], device=dev)
                    h = net.encoder_backbone(xb)
                    mu, lv = net.fc_mu(h), net.fc_logvar(h)
                    mse = ((xb - net.decoder(mu)) ** 2).mean(1)
                    kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).sum(1)
                    mus.append(mu.cpu().numpy())
                    scores.append((mse + VAE_KL_WEIGHT * kl).cpu().numpy())
            return Xs, np.concatenate(mus), np.concatenate(scores)
        _ENC[key] = forward
    return _ENC[key]


# ------------------------------------------------------------------ loader: (root, thu muc, fold) -> score(X)
def _ck(root, d, fold):
    return root / d / fold / "checkpoint"


def scaled_model(root, d, fold):                 # XGBoost-only, LightGBM: scaler rieng, khong cat
    m, s = joblib.load(_ck(root, d, fold) / "model.pkl"), joblib.load(_ck(root, d, fold) / "scaler.pkl")
    return lambda X: _proba(m, s.transform(X))


class PyTorchKNNClassifier:
    """Thay cho lop k-NN cua baseline study (pickle duoi __main__): vote theo 1/khoang cach cua k lang gieng."""

    def predict_proba(self, X, batch=1024):
        import torch
        Xt = torch.from_numpy(np.ascontiguousarray(self.X_train_cpu, dtype=np.float32))
        yt = torch.from_numpy(np.asarray(self.y_train_cpu, dtype=np.float32))
        p1 = np.empty(len(X), dtype=np.float32)
        for i in range(0, len(X), batch):
            q = torch.from_numpy(np.ascontiguousarray(X[i:i + batch], dtype=np.float32))
            dk, ik = torch.cdist(q, Xt).topk(self.n_neighbors, largest=False)
            w = 1.0 / dk.clamp_min(1e-12) if self.weights == "distance" else torch.ones_like(dk)
            p1[i:i + batch] = ((w * yt[ik]).sum(1) / w.sum(1)).numpy()
        return np.stack([1 - p1, p1], axis=1)


def knn(root, d, fold):
    sys.modules["__main__"].PyTorchKNNClassifier = PyTorchKNNClassifier   # ten lop trong pickle
    m, s = joblib.load(_ck(root, d, fold) / "model.pkl"), joblib.load(_ck(root, d, fold) / "scaler.pkl")
    return lambda X: m.predict_proba(s.transform(X))[:, 1]


def random_forest(root, d, fold):
    m = joblib.load(_ck(root, d, fold) / "model_sklearn_cpu.pkl")
    return lambda X: _proba(m, X)


def svm(root, d, fold):
    m = joblib.load(_ck(root, d, fold) / "model.pkl")
    return lambda X: m.decision_function(X)


def ae_features(root):                           # [78 | latent 16 | loi tai tao]
    fwd = ae_encoder(root)

    def feats(X):
        Xs, z, e = fwd(X)
        return np.concatenate([Xs, z, e[:, None]], 1)
    return feats


def svdd_features(root):                         # [78 | K khoang cach sphere | d_min]
    fwd = svdd_encoder(root)

    def feats(X):
        Xs, dist = fwd(X)
        return np.concatenate([Xs, dist, dist.min(1, keepdims=True)], 1)
    return feats


def vae_features(root):                          # [78 | mu 32 | diem VAE]
    fwd = vae_encoder(root)

    def feats(X):
        Xs, mu, s = fwd(X)
        return np.concatenate([Xs, mu, s[:, None]], 1)
    return feats


def hybrid(features):
    """Loader cua model hybrid: XGBoost tren features(root)(X)."""
    def load(root, d, fold):
        m, feats = joblib.load(_ck(root, d, fold) / "model.pkl"), features(root)
        return lambda X: _proba(m, feats(X))
    return load


def clad(root, d, fold):                         # -cos(z, tam benign)
    paths.use_layer1_code()
    from models.clad.model import build_model
    from models.clad.util.distance import cosine_scores_to_centroid
    torch, dev = _torch()
    ck = _ck(root, d, fold)
    sc = joblib.load(ck / "scaler.pkl")
    meta = json.loads((ck / "meta.json").read_text(encoding="utf-8"))
    net = build_model(d_in=N_FEATURES, hidden_dim=512, n_layers=4, d_out=32, dropout=0.0, block_norm=False).to(dev)
    net.load_state_dict(torch.load(ck / "clad_best.pt", map_location=dev, weights_only=False))
    net.eval()
    cen = torch.tensor(meta["centroid"], dtype=torch.float32)

    def score(X):
        Xs = np.clip(sc.transform(X), -CLIP, CLIP).astype("float32")
        zl = []
        with torch.no_grad():
            for i in range(0, len(Xs), 8192):
                zl.append(net(torch.tensor(Xs[i:i + 8192], device=dev)).cpu())
        return np.asarray(-cosine_scores_to_centroid(torch.cat(zl), cen))
    return score


def ae_only(root, d, fold):
    fwd = ae_encoder(root)
    return lambda X: fwd(X)[2]


def svdd_only(root, d, fold):
    fwd = svdd_encoder(root)
    return lambda X: fwd(X)[1].min(1)


def deep_svdd(root, d, fold):
    return deep_svdd_encoder(root)


def vae_only(root, d, fold):
    fwd = vae_encoder(root)
    return lambda X: fwd(X)[2]


def optiforest(root, d, fold):
    """OptIForest (Xiang et al., IJCAI 2023): code cua tac gia (package detectors/) can de unpickle.
    Diem = -decision_function (giong demo.py cua tac gia), sau scaler + cat [-20, 20]."""
    key = ("optiforest", str(root))
    if key not in _ENC:
        if str(paths.OPTIFOREST_CODE) not in sys.path:
            sys.path.insert(0, str(paths.OPTIFOREST_CODE))
        import detectors  # noqa: F401  (lop cua model trong pickle)
        ck = root / d / "checkpoint"
        sc, m = joblib.load(ck / "scaler.pkl"), joblib.load(ck / "optiforest_model.pkl")

        def score(X):
            Xs = np.clip(sc.transform(X), -CLIP, CLIP)
            return np.concatenate([-1.0 * m.decision_function(Xs[i:i + 20000]) for i in range(0, len(Xs), 20000)])
        _ENC[key] = score
    return _ENC[key]


# ------------------------------------------------------------------ danh sach model
@dataclass(frozen=True)
class Spec:
    label: str            # ten trong bang
    family: str           # Supervised / One-class / Contrastive / Hybrid
    per_fold: bool        # moi fold mot checkpoint (True) hay mot checkpoint cho ca 14 fold (False)
    load: Callable        # load(root, dir, fold) -> score(X)
    dir: str              # thu muc checkpoint duoi root


MODELS = {
    "xgboost_only":      Spec("XGBoost-only", "Supervised", True, scaled_model, "xgboost_only"),
    "lgbm":              Spec("LightGBM", "Supervised", True, scaled_model, "lgbm"),
    "random_forest":     Spec("Random Forest", "Supervised", True, random_forest, "random_forest"),
    "knn":               Spec("KNN", "Supervised", True, knn, "knn"),
    "svm_only":          Spec("SVM (linear)", "Supervised", True, svm, "svm_only"),
    "ae_only":           Spec("AE-only", "One-class", False, ae_only, "ae_only"),
    "svdd_only":         Spec("Mixture Deep SVDD", "One-class", False, svdd_only, "svdd_only"),
    "deep_svdd":         Spec("Deep SVDD", "One-class", False, deep_svdd, "deep_svdd_paper"),
    "vae_only":          Spec("VAE-only", "One-class", False, vae_only, "vae_only"),
    "optiforest":        Spec("OptiForest", "One-class", False, optiforest, "optiforest"),
    "clad":              Spec("CLAD", "Contrastive", True, clad, "clad"),
    "ae_xgboost":        Spec("AE-XGBoost", "Hybrid", True, hybrid(ae_features), "ae_xgboost"),
    "svdd_xgboost":      Spec("SVDD-XGBoost", "Hybrid", True, hybrid(svdd_features), "svdd_xgboost"),
    "vae_xgboost":       Spec("VAE-XGBoost", "Hybrid", True, hybrid(vae_features), "vae_xgboost"),
    "ae_xgboost_mono":   Spec("AE-XGBoost (ours)", "Hybrid", True, hybrid(ae_features), "ae_xgboost_mono"),
    "svdd_xgboost_mono": Spec("SVDD-XGBoost (ours)", "Hybrid", True, hybrid(svdd_features), "svdd_xgboost_mono"),
}
HYBRID_FEATURES = {"ae_xgboost": ae_features, "ae_xgboost_mono": ae_features, "svdd_xgboost": svdd_features,
                   "svdd_xgboost_mono": svdd_features, "vae_xgboost": vae_features}


def loader(key, source, fold):
    """score(X) cua model `key` (checkpoint cua `source`) o fold `fold`."""
    spec = MODELS[key]
    return spec.load(ROOT[source], spec.dir, fold)


def tree_model(key, source, fold):
    """(model cay cuoi cung, X tho -> input cua model) de chay TreeSHAP; (None, None) neu khong phai model cay."""
    root = ROOT[source]
    ck = _ck(root, MODELS[key].dir, fold)
    if key in ("xgboost_only", "lgbm"):
        return joblib.load(ck / "model.pkl"), joblib.load(ck / "scaler.pkl").transform
    if key == "random_forest":
        return joblib.load(ck / "model_sklearn_cpu.pkl"), (lambda X: X)
    if key in HYBRID_FEATURES:
        return joblib.load(ck / "model.pkl"), HYBRID_FEATURES[key](root)
    return None, None
