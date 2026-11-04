"""
Layer 1: X tho (78 feature) -> xac suat tan cong p(x) va vector SHAP. Day la noi DUY NHAT tinh SHAP
(pipeline, tap R ca hai protocol, bundle Setup 1/2, demo live).

Hai bo checkpoint, cung mot giao dien (build_input, predict_proba, shap_values, score):
    Layer1Engine(variant)                                Protocol B: <UTS_DATA>/artifacts/layer1/<variant>/
                                                         (layer1/train.py tren train_full.csv)
    Layer1Engine.from_fold(variant, fold, root, base)    Protocol A: checkpoint 14 fold LOAO cua LAYER1
                                                         (zeroshot_benchmark/04_training_v2/artifacts)

SHAP = shap.TreeExplainer tren XGBoost cuoi cung (TreeSHAP chinh xac, path-dependent). Voi XGBoost, shap goi
pred_contribs cua chinh xgboost, nen ket qua trung tung bit voi booster.predict(pred_contribs=True)[:, :-1]
ma cac tap R Protocol A da dung (da kiem tren ca hai protocol, shap 0.51, xgboost 3.2).
"""
import json
import warnings
from pathlib import Path

import numpy as np

import config

AE_BOTTLENECK = 16


class Layer1Engine:
    """XGBoost tang cuong + TreeExplainer.

    Protocol B: duong di phai khop CHINH XAC voi layer1/train.py, neu khong model se
    nhin thay du lieu khac luc train:

        X tho (78) -> RobustScaler -> clip(+-20) -> [78 | feature tang cuong]
            AE   : 16 latent + 1 recon_error   -> 95 chieu
            SVDD : K sphere_dist + 1 d_min     -> 78+K+1 chieu

    Protocol A (from_fold): nhu luc train 14 fold cua LAYER1 - scaler rieng cua encoder
    (ae_only / svdd_only), XGBoost-only dung scaler cua fold va khong clip.

    TreeExplainer chay tren vector day du, nen SHAP tra ve cung so chieu. Vector
    do la khoa truy hoi cho retriever: chieu augmented (recon_error, d_min) mang tin
    hieu "lech khoi benign".
    """

    def __init__(self, variant=None):
        import joblib

        variant = variant or config.VARIANT
        art = config.LAYER1_ARTIFACTS[variant]

        self.variant = variant
        self.fold = None
        self.model = joblib.load(art["model"])
        self.scaler = joblib.load(art["scaler"])
        self.threshold = config.layer1_threshold(variant)

        with open(art["features"], encoding="utf-8") as f:
            self.feature_names = json.load(f)
        self.n_features = int(getattr(self.model, "n_features_in_",
                                      len(self.feature_names)))
        self.n_base = int(self.scaler.n_features_in_)

        self._load_encoder(art["encoder"])
        self._init_explainer()

        print(f"Layer 1 ready: {variant} | {self.n_base} base + "
              f"{self.n_features - self.n_base} augmented = {self.n_features} "
              f"feature | threshold={self.threshold:.4f}")

    @classmethod
    def from_fold(cls, variant, fold, root, base_features, threshold=None, device=None, xgb_device=None):
        """Checkpoint Protocol A cua mot fold leave-one-attack-out.

        Args:
            variant: "xgboost_only" | "ae_xgboost_mono" | "svdd_xgboost_mono" (thu muc checkpoint).
            fold: loai tan cong bi giu lai (ten thu muc fold).
            root: thu muc artifacts cua LAYER1 (experiments/paths.L1_ART).
            base_features: 78 ten cot feature goc, theo thu tu trong CSV.
            threshold: tau_alpha cua fold (vd tu meta cua tap R); None neu chua can.
            device: thiet bi cua encoder AE / SVDD; mac dinh cuda neu co.
            xgb_device: thiet bi cua XGBoost (predict va SHAP); mac dinh = device.
        """
        import joblib
        import torch

        from layer1.models.autoencoder import BenignAutoencoder
        from layer1.models.svdd import MixtureSVDD

        self = cls.__new__(cls)
        self.variant, self.fold, self.threshold = variant, fold, threshold
        self.torch = torch
        # CUDA_VISIBLE_DEVICES="" van de is_available() = True nhung khong co thiet bi nao
        gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
        self.device = torch.device(device or ("cuda" if gpu else "cpu"))
        root, base = Path(root), list(base_features)

        self.model = joblib.load(root / variant / fold / "checkpoint" / "model.pkl")
        self.model.get_booster().set_param({"device": xgb_device or self.device.type})
        self.ae = self.svdd = None
        if variant == "xgboost_only":
            self.scaler = joblib.load(root / variant / fold / "checkpoint" / "scaler.pkl")
            aug = []
        elif variant.startswith("ae"):
            enc = root / "ae_only" / "checkpoint"
            self.scaler = joblib.load(enc / "scaler.pkl")
            self.ae = BenignAutoencoder(input_dim=len(base), bottleneck_dim=AE_BOTTLENECK).to(self.device)
            self.ae.load_state_dict(torch.load(enc / "ae_best.pt", map_location=self.device))
            self.ae.eval()
            aug = [f"ae_latent_{i}" for i in range(AE_BOTTLENECK)] + ["ae_recon_error"]
        else:
            enc = root / "svdd_only" / "checkpoint"
            self.scaler = joblib.load(enc / "scaler.pkl")
            k = len(list((enc / "svdd_model").glob("encoder_*.pt")))
            self.svdd = MixtureSVDD(K=k, input_dim=len(base), embed_dim=64, device=self.device)
            self.svdd.load(str(enc / "svdd_model"))
            aug = [f"svdd_sphere_{i}" for i in range(k)] + ["svdd_d_min"]

        self.feature_names = base + aug
        self.n_base, self.n_features = len(base), len(base) + len(aug)
        self._init_explainer()
        return self

    # ------------------------------------------------------------------
    def _init_explainer(self):
        import shap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.explainer = shap.TreeExplainer(self.model)

    def _load_encoder(self, path):
        """Protocol B: nap AE hoac SVDD tuy variant."""
        self.ae = self.svdd = None

        if self.variant.startswith("ae"):
            import torch
            from layer1.models.autoencoder import BenignAutoencoder
            n_aug = self.n_features - self.n_base          # 16 latent + 1 error
            self.ae = BenignAutoencoder(input_dim=self.n_base,
                                        bottleneck_dim=n_aug - 1)
            self.ae.load_state_dict(torch.load(path, map_location="cpu",
                                               weights_only=True))
            self.ae.eval()
            self.torch = torch
        else:
            from layer1.models.svdd import MixtureSVDD
            K = self.n_features - self.n_base - 1          # K sphere + 1 d_min
            self.svdd = MixtureSVDD(K=K, input_dim=self.n_base, device="cpu")
            self.svdd.load(path)

    def base_feature_columns(self, df):
        """78 cot feature goc, bo metadata, giu nguyen thu tu trong CSV."""
        cols = [c for c in df.columns if c not in config.META_COLS]
        if len(cols) != self.n_base:
            raise ValueError(
                f"So feature goc khong khop: CSV co {len(cols)}, scaler can "
                f"{self.n_base}. Kiem tra META_COLS trong config.py."
            )
        return cols

    # Giu ten cu de code goi truoc do khong hong
    feature_columns = base_feature_columns

    def preprocess(self, X):
        """Protocol B: X tho -> scale -> clip. Khop layer1/train.py."""
        X = np.nan_to_num(np.asarray(X, dtype="float64"),
                          nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(self.scaler.transform(X), -config.CLIP_VAL, config.CLIP_VAL)

    def augment(self, Xs, batch=8192):
        """Protocol B: Xs da scale -> ma tran feature tang cuong."""
        if self.ae is not None:
            outs = []
            with self.torch.no_grad():
                for i in range(0, len(Xs), batch):
                    xb = self.torch.tensor(Xs[i:i + batch], dtype=self.torch.float32)
                    xh, z = self.ae(xb)
                    err = ((xb - xh) ** 2).mean(dim=1, keepdim=True)
                    outs.append(self.torch.cat([z, err], dim=1).numpy())
            return np.vstack(outs)

        d = self.svdd.all_sphere_distances(Xs)
        return np.hstack([d, d.min(axis=1, keepdims=True)])

    def _build_input_fold(self, X, batch=8192):
        """Protocol A: giong luc build tap R tung fold (01_build_loao_pools)."""
        if self.ae is None and self.svdd is None:                  # xgboost_only: khong clip
            return self.scaler.transform(X)
        Xs = np.clip(self.scaler.transform(X), -config.CLIP_VAL, config.CLIP_VAL)
        if self.ae is not None:
            Xs = Xs.astype("float32")
            zs, es = [], []
            with self.torch.no_grad():
                for i in range(0, len(Xs), batch):
                    xb = self.torch.tensor(Xs[i:i + batch], device=self.device)
                    xh, z = self.ae(xb)
                    zs.append(z.cpu().numpy())
                    es.append(((xb - xh) ** 2).mean(1).cpu().numpy())
            return np.concatenate([Xs, np.concatenate(zs), np.concatenate(es)[:, None]], 1)
        d = self.svdd.all_sphere_distances(Xs)
        return np.concatenate([Xs, d, d.min(1, keepdims=True)], 1)

    def build_input(self, X):
        """X tho -> ma tran dung cho XGBoost (da scale, clip, ghep augmented)."""
        if self.fold is not None:
            return self._build_input_fold(X)
        Xs = self.preprocess(X)
        return np.hstack([Xs, self.augment(Xs)])

    def predict_proba(self, X):
        """Xac suat tan cong tu feature THO."""
        return self.model.predict_proba(self.build_input(X))[:, 1]

    def shap_values(self, A):
        """TreeSHAP chinh xac tren input cua model (ket qua cua build_input)."""
        sv = self.explainer.shap_values(A)
        if isinstance(sv, list):          # mot so phien ban tra list theo class
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                  # (n, feat, class)
            sv = sv[:, :, 1]
        return sv

    def score(self, X):
        """Tra (proba_attack, shap_values) cho ma tran feature THO.

        SHAP co cung so chieu voi input cua model (95 voi AE, 78+K+1 voi SVDD).
        """
        Xf = self.build_input(X)
        return self.model.predict_proba(Xf)[:, 1], self.shap_values(Xf)
