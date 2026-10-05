"""Registry of the extended Layer 1 baselines (Protocol A, 14 leave-one-attack-out folds).

Each entry turns a trained checkpoint into a scoring function
    score(X) -> (n,) attack score, higher = attack,
where X holds the 78 raw CICFlowMeter features of n flows. Scoring and latency scripts
both use this registry, so a model is loaded and preprocessed the same way everywhere.

Checkpoints live under paths.BASE_ART/<model>/ (env UTS_BASELINE_ART), in the layout
produced by the baseline training scripts:
    <model>/<fold>/checkpoint/{model.pkl, scaler.pkl, ...}   per-fold supervised models
    <model>/checkpoint/...                                    one-class models (one per study)
    <model>/metrics.json                                      the trainer's own evaluation

`own_threshold(model, metrics, fold)` returns the threshold and comparison each trainer
used, so 02_verify_reproduction.py can confirm the checkpoints are read correctly.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

CLIP = 20.0
N_FEATURES = 78


@dataclass(frozen=True)
class Spec:
    label: str            # name in tables
    family: str           # Supervised / One-class / Contrastive / Hybrid
    per_fold: bool        # one checkpoint per fold (True) or one for all folds (False)
    loader: Callable      # loader(fold) -> score(X)
    tree: bool = False    # exact TreeSHAP available (latency of decision + attribution)


def _ck(model, fold):
    return paths.BASE_ART / model / fold / "checkpoint"


def _proba(m, A):
    pos = int(np.flatnonzero(m.classes_ == 1)[0]) if hasattr(m, "classes_") else 1
    return m.predict_proba(A)[:, pos]


def _torch():
    """torch and the device to use; CPU when no GPU is visible (e.g. CUDA_VISIBLE_DEVICES="")."""
    import torch
    gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    return torch, torch.device("cuda" if gpu else "cpu")


# ------------------------------------------------------------------ shared encoders
_ENC = {}


def _ae():
    """Benign-only autoencoder of the baseline study: X -> (X_scaled, latent, recon error)."""
    if "ae" not in _ENC:
        paths.use_layer1_code()
        from models.autoencoder import BenignAutoencoder
        torch, dev = _torch()
        d = paths.BASE_ART / "ae_only" / "checkpoint"
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
        _ENC["ae"] = forward
    return _ENC["ae"]


def _svdd():
    """Mixture Deep SVDD of the baseline study: X -> (X_scaled, sphere distances)."""
    if "svdd" not in _ENC:
        paths.use_layer1_code()
        from models.svdd import MixtureSVDD
        _, dev = _torch()
        d = paths.BASE_ART / "svdd_only" / "checkpoint"
        sc = joblib.load(d / "scaler.pkl")
        k_in = len(list((d / "svdd_model").glob("encoder_*.pt")))
        sv = MixtureSVDD(K=k_in, input_dim=N_FEATURES, embed_dim=64, device=str(dev))
        sv.load(str(d / "svdd_model"))

        def forward(X):
            Xs = np.clip(sc.transform(X), -CLIP, CLIP)
            return Xs, sv.all_sphere_distances(Xs)
        _ENC["svdd"] = forward
    return _ENC["svdd"]


def _vae_net(d_in=N_FEATURES, hidden=(512, 256), latent=32, slope=0.2, p=0.1):
    """Benign VAE of the baseline study. Rebuilt from vae_best.pt (parameter names and shapes)
    and vae_params in vae_only/metrics.json: Linear -> LayerNorm -> LeakyReLU -> Dropout blocks."""
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


# Anomaly score of the VAE: the per-feature VAE loss (SSE + beta * KL) / 78, i.e. reconstruction
# MSE (decoded from mu) + (beta / 78) * KL summed over latent dimensions, beta = 0.1 (vae_params).
# This form reproduces the trainer's reported metrics exactly (step 2); MSE alone or
# beta * KL / 32 do not.
VAE_KL_WEIGHT = 0.1 / N_FEATURES


def _vae():
    """X -> (X_scaled, mu, anomaly score)."""
    if "vae" not in _ENC:
        torch, dev = _torch()
        d = paths.BASE_ART / "vae_only" / "checkpoint"
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
        _ENC["vae"] = forward
    return _ENC["vae"]


# ------------------------------------------------------------------ loaders
def _scaled_model(name):
    def load(fold):
        m, s = joblib.load(_ck(name, fold) / "model.pkl"), joblib.load(_ck(name, fold) / "scaler.pkl")
        return lambda X: _proba(m, s.transform(X))
    return load


class PyTorchKNNClassifier:
    """Stand-in for the baseline study's k-NN class, which was pickled under `__main__`.

    Rebuilt from the attributes the pickle stores: X_train_cpu (50K scaled training flows),
    y_train_cpu, n_neighbors = 5, weights = "distance". P(attack) is the inverse-distance
    weighted vote of the k nearest training flows (Euclidean); 02_verify_reproduction.py
    checks it against the trainer's metrics.
    """

    def predict_proba(self, X, batch=1024):   # 1024 x 50K distances ~ 200 MB per batch
        import torch
        Xt = torch.from_numpy(np.ascontiguousarray(self.X_train_cpu, dtype=np.float32))
        yt = torch.from_numpy(np.asarray(self.y_train_cpu, dtype=np.float32))
        p1 = np.empty(len(X), dtype=np.float32)
        for i in range(0, len(X), batch):
            q = torch.from_numpy(np.ascontiguousarray(X[i:i + batch], dtype=np.float32))
            dk, ik = torch.cdist(q, Xt).topk(self.n_neighbors, largest=False)
            if self.weights == "distance":
                w = 1.0 / dk.clamp_min(1e-12)
            else:
                w = torch.ones_like(dk)
            p1[i:i + batch] = ((w * yt[ik]).sum(1) / w.sum(1)).numpy()
        return np.stack([1 - p1, p1], axis=1)


def _knn(fold):
    sys.modules["__main__"].PyTorchKNNClassifier = PyTorchKNNClassifier   # class named in the pickle
    m = joblib.load(_ck("knn", fold) / "model.pkl")
    s = joblib.load(_ck("knn", fold) / "scaler.pkl")
    return lambda X: m.predict_proba(s.transform(X))[:, 1]


def _random_forest(fold):
    m = joblib.load(_ck("random_forest", fold) / "model_sklearn_cpu.pkl")
    return lambda X: _proba(m, X)


def _svm(fold):
    m = joblib.load(_ck("svm_only", fold) / "model.pkl")
    return lambda X: m.decision_function(X)


def _ae_xgboost(fold):
    m = joblib.load(_ck("ae_xgboost", fold) / "model.pkl")
    fwd = _ae()

    def score(X):
        Xs, z, e = fwd(X)
        return _proba(m, np.concatenate([Xs, z, e[:, None]], 1))
    return score


def _svdd_xgboost(fold):
    m = joblib.load(_ck("svdd_xgboost", fold) / "model.pkl")
    fwd = _svdd()

    def score(X):
        Xs, d = fwd(X)
        return _proba(m, np.concatenate([Xs, d, d.min(1, keepdims=True)], 1))
    return score


def _clad(fold):
    paths.use_layer1_code()
    from models.clad.model import build_model
    from models.clad.util.distance import cosine_scores_to_centroid
    torch, dev = _torch()
    d = _ck("clad", fold)
    sc = joblib.load(d / "scaler.pkl")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    net = build_model(d_in=N_FEATURES, hidden_dim=512, n_layers=4, d_out=32, dropout=0.0,
                      block_norm=False).to(dev)
    net.load_state_dict(torch.load(d / "clad_best.pt", map_location=dev, weights_only=False))
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


def _ae_only(fold):
    fwd = _ae()
    return lambda X: fwd(X)[2]


def _svdd_only(fold):
    fwd = _svdd()
    return lambda X: fwd(X)[1].min(1)


def _vae_only(fold):
    fwd = _vae()
    return lambda X: fwd(X)[2]


def _vae_xgboost(fold):
    m = joblib.load(_ck("vae_xgboost", fold) / "model.pkl")
    fwd = _vae()

    def score(X):
        Xs, mu, s = fwd(X)
        return _proba(m, np.concatenate([Xs, mu, s[:, None]], 1))   # [78 | 32 latent | score], 111 columns
    return score


REGISTRY = {
    "xgboost_only":  Spec("XGBoost-only", "Supervised", True, _scaled_model("xgboost_only"), tree=True),
    "lgbm":          Spec("LightGBM", "Supervised", True, _scaled_model("lgbm"), tree=True),
    "random_forest": Spec("Random Forest", "Supervised", True, _random_forest, tree=True),
    "knn":           Spec("KNN", "Supervised", True, _knn),
    "svm_only":      Spec("SVM (linear)", "Supervised", True, _svm),
    "ae_only":       Spec("AE-only", "One-class", False, _ae_only),
    "svdd_only":     Spec("Mixture Deep SVDD", "One-class", False, _svdd_only),
    "vae_only":      Spec("VAE-only", "One-class", False, _vae_only),
    "clad":          Spec("CLAD", "Contrastive", True, _clad),
    "ae_xgboost":    Spec("AE-XGBoost", "Hybrid", True, _ae_xgboost, tree=True),
    "svdd_xgboost":  Spec("SVDD-XGBoost", "Hybrid", True, _svdd_xgboost, tree=True),
    "vae_xgboost":   Spec("VAE-XGBoost", "Hybrid", True, _vae_xgboost, tree=True),
}


def folds():
    return sorted(p.name for p in paths.SPLITS.iterdir() if p.is_dir())


# Files each model reads; "{fold}" expands to every fold. Hybrids also need their encoder.
_AE_FILES = ["ae_only/checkpoint/scaler.pkl", "ae_only/checkpoint/ae_best.pt"]
_SVDD_FILES = ["svdd_only/checkpoint/scaler.pkl", "svdd_only/checkpoint/svdd_model/centers.npy",
               "svdd_only/checkpoint/svdd_model/encoder_0.pt"]
REQUIRED = {
    "xgboost_only": ["xgboost_only/{fold}/checkpoint/model.pkl", "xgboost_only/{fold}/checkpoint/scaler.pkl"],
    "lgbm": ["lgbm/{fold}/checkpoint/model.pkl", "lgbm/{fold}/checkpoint/scaler.pkl"],
    "knn": ["knn/{fold}/checkpoint/model.pkl", "knn/{fold}/checkpoint/scaler.pkl"],
    "random_forest": ["random_forest/{fold}/checkpoint/model_sklearn_cpu.pkl"],
    "svm_only": ["svm_only/{fold}/checkpoint/model.pkl"],
    "ae_only": _AE_FILES,
    "svdd_only": _SVDD_FILES,
    "vae_only": ["vae_only/checkpoint/vae_best.pt", "vae_only/checkpoint/scaler.pkl"],
    "clad": ["clad/{fold}/checkpoint/clad_best.pt", "clad/{fold}/checkpoint/meta.json",
             "clad/{fold}/checkpoint/scaler.pkl"],
    "ae_xgboost": ["ae_xgboost/{fold}/checkpoint/model.pkl"] + _AE_FILES,
    "svdd_xgboost": ["svdd_xgboost/{fold}/checkpoint/model.pkl"] + _SVDD_FILES,
    "vae_xgboost": ["vae_xgboost/{fold}/checkpoint/model.pkl", "vae_only/checkpoint/vae_best.pt",
                    "vae_only/checkpoint/scaler.pkl"],
}


def available(name):
    """True if every file this model reads is on disk."""
    for rel in REQUIRED[name]:
        for f in (folds() if "{fold}" in rel else [None]):
            if not (paths.BASE_ART / rel.format(fold=f)).exists():
                return False
    return True


def own_threshold(name, metrics, fold):
    """(threshold, strict) as used by the trainer itself; strict=True means score > t."""
    if "threshold_mean_used_for_eval" in metrics:                 # XGBoost family, LightGBM, KNN
        return metrics["threshold_mean_used_for_eval"], False
    if name == "clad":                                             # F1-max on val, per fold
        meta = json.loads((_ck("clad", fold) / "meta.json").read_text(encoding="utf-8"))
        return meta["threshold"], False
    for k in ("threshold_per_label_from_benign_train", "threshold_per_label"):
        if k in metrics:
            return metrics[k][fold], True
    for k in ("ae_threshold", "vae_threshold", "threshold_benign_train", "svdd_threshold"):
        if k in metrics:
            return metrics[k], True
    raise KeyError(f"{name}: no threshold field in metrics.json")
