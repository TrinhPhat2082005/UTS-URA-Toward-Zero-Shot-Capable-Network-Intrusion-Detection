"""
[BAN _mono] Train AE+XGBoost VOI RANG BUOC DON DIEU tren ae_recon_error.

Khac ban goc train_ae_xgboost.py DUNG 3 diem (moi thu khac giu nguyen 100% de
so sanh sach — cung data v2, cung AE encoder, cung threshold-averaging):
  1. Import config tu common/xgb_config_mono.py (thay vi xgb_config.py)
  2. Them monotone_constraints: CHI ae_recon_error bi ep DON DIEU TANG
     (recon_error tang => P(attack) khong duoc giam). 16 ae_latent va 78 feature
     goc giu TU DO — latent la toa do khong gian nen, khong co ngu nghia don dieu.
  3. Ghi ra artifacts/ae_xgboost_mono/ (khong de len ket qua cu)

LY DO: tren data v2, AE+XGBoost (ZS-F1 0.574) con THAP HON XGBoost-only (0.586)
du SHAP xac nhan ae_recon_error XEP HANG 1 (0.79, vuot ca Dst Port 0.78).
Nguyen nhan: XGBoost hoc LUAT LIEN HOP "recon_error cao NHUNG cac feature goc
trong quen => benign", lam mat tin hieu one-class tren attack chua tung thay.
Xem docstring common/xgb_config_mono.py de biet chi tiet co che va danh doi.

--- Docstring goc ---
Train AE+XGBoost (augmented) — train RIENG 14 lan, moi lan 1 held-out label.
Tai su dung AE checkpoint DA TRAIN SAN o artifacts/ae_only/checkpoint/ (khong train
lai AE — AE chi hoc Benign 1 lan duy nhat, dung chung cho ca 14 fold, dung thiet ke
da chot voi AE-Only). AE CHI dong vai tro feature encoder: latent_z (16) + recon_error
(1) duoc noi vao 78 feature goc = 95 feature cho XGBoost. XGBoost la diem quyet dinh
DUY NHAT — khong co nguong AE rieng, khong OR-Fusion (dung thiet ke notebook goc
UTS_Layer1/AE_XGBoost/autoencoder_xgboost_ids_v2.ipynb).

DUNG CHUNG config XGBoost voi train_xgboost_only.py (common/xgb_config.py) de so
sanh cong bang — chi khac o feature input (78 vs 95).

Pipeline (moi held-out label):
  1. Doc train_benign_attack.csv/val_binary.csv/test_zeroshot.csv cua label do.
  2. Dung CHUNG 1 scaler (ae_scaler, fit tren benign-only, tai su dung tu
     artifacts/ae_only/) cho CA XGBoost input LAN AE encoder input — dung dung
     pattern notebook goc UTS_Layer1/AE_XGBoost/autoencoder_xgboost_ids_v2.ipynb
     (Cell 3+11: 1 scaler.transform() duy nhat cho ca X_ae_train_s va X_xgb_train_s,
     KHONG co scaler rieng cho XGBoost). Ban truoc dung 2 scaler khac nhau (mot fit
     tren train_benign_attack, mot fit tren benign-only) cho cung 1 tap X_train_raw —
     loi kien truc tuong tu da phat hien va sua trong train_svdd_xgboost.py, khien
     AE-encoder va XGBoost nhin thay 2 phien ban du lieu qua bien doi phan phoi khac
     nhau.
  3. Dung AE (da load, eval mode) tinh latent_z + recon_error tren CHINH du lieu da
     scale o buoc 2 (khong con 2 phien ban rieng biet nua).
  4. Ghep [X_scaled | latent_z | recon_error] = 95 feature, train XGBClassifier.
  5. Tune threshold F1-max tren TRAIN (train_benign_attack.csv), khong phai val —
     vi val_binary.csv la tap DUNG DE BAO CAO known-attack F1/recall/FPR; tune
     threshold tren chinh tap eval se lam ket qua bi thoi phong (data leak kieu
     "threshold biet truoc dap an"). Sau khi train xong CA 14 fold, lay TRUNG
     BINH 14 threshold nay thanh 1 threshold duy nhat, roi dung LAI threshold
     chung do de eval lai toan bo 14 fold (known-attack + zero-shot) — mo phong
     dung mot nguong quyet dinh co dinh nhu khi trien khai thuc te.
  6. Luu checkpoint (model.pkl) vao artifacts/ae_xgboost/<label>/checkpoint/ (khong
     con scaler rieng — dung chung ae_scaler tu artifacts/ae_only/).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.xgb_config_mono import (  # noqa: E402
    XGB_PARAMS,
    XGB_SCALE_POS_DIVISOR,
    XGB_MONOTONE_AUGMENTED,
    build_monotone_constraints,
)
from models.autoencoder import BenignAutoencoder  # noqa: E402
from models.clad.util.metrics import best_f1_threshold  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = BASE_DIR / "03_splits_v2"
LABELS_CONFIG_PATH = BASE_DIR / "config" / "labels.json"

AE_ONLY_CKPT_DIR = Path(__file__).resolve().parent / "artifacts" / "ae_only" / "checkpoint"
AE_SCALER_PATH = AE_ONLY_CKPT_DIR / "scaler.pkl"
AE_MODEL_PATH = AE_ONLY_CKPT_DIR / "ae_best.pt"

ARTIFACTS_ROOT = Path(__file__).resolve().parent / "artifacts" / "ae_xgboost_mono"
METRICS_PATH = ARTIFACTS_ROOT / "metrics.json"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BOTTLENECK_DIM = 16
CLIP_VAL = 20.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XGB_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_split_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "Label" in df.columns and "Label_Text" in df.columns, f"{path}: thieu cot Label/Label_Text"
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("Label", "Label_Text")]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tra ve dung 6 chi so: f1, recall, precision, accuracy, fpr, n_samples."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(rec),
        "precision": float(pre),
        "accuracy": float(acc),
        "fpr": float(fpr),
        "n_samples": int(len(y_true)),
    }


def get_ae_features(ae_model: BenignAutoencoder, X_ae_scaled: np.ndarray, batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Tra ve (latent_z, recon_error) cho input DA QUA AE'S SCALER (khac scaler XGB)."""
    ae_model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X_ae_scaled, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    latents, errors = [], []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            x_hat, z = ae_model(xb)
            err = ((xb - x_hat) ** 2).mean(dim=1)
            latents.append(z.cpu().numpy())
            errors.append(err.cpu().numpy())
    return np.concatenate(latents), np.concatenate(errors)


def train_one_label(label: str, feature_cols: list[str], ae_model: BenignAutoencoder, ae_scaler) -> dict:
    label_artifacts_dir = ARTIFACTS_ROOT / label
    images_dir = label_artifacts_dir / "images"
    ckpt_dir = label_artifacts_dir / "checkpoint"
    images_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_path = ckpt_dir / "model.pkl"

    train_df = load_split_csv(SPLITS_DIR / "by_label" / label / "train_benign_attack.csv")
    val_df = load_split_csv(SPLITS_DIR / "by_label" / label / "val_binary.csv")
    test_df = load_split_csv(SPLITS_DIR / "by_label" / label / "test_zeroshot.csv")

    X_train_raw = train_df[feature_cols].values.astype("float32")
    y_train = train_df["Label"].values
    X_val_raw = val_df[feature_cols].values.astype("float32")
    y_val = val_df["Label"].values
    X_test_raw = test_df[feature_cols].values.astype("float32")
    y_test = test_df["Label"].values

    # Dung CHUNG 1 scaler (ae_scaler, fit tren benign-only, tai su dung tu
    # artifacts/ae_only/) cho CA XGBoost input LAN AE encoder input — dung dung
    # pattern notebook goc UTS_Layer1/AE_XGBoost Cell 3+11 (mot scaler.transform() duy
    # nhat cho ca X_ae_train_s va X_xgb_train_s, KHONG co scaler rieng cho XGBoost).
    # Ban truoc dung 2 scaler khac nhau (scaler_xgb fit tren train_benign_attack,
    # ae_scaler fit tren benign-only) cho cung 1 tap X_train_raw — loi kien truc
    # tuong tu da phat hien va sua trong train_svdd_xgboost.py.
    X_train_xgb_s = np.clip(ae_scaler.transform(X_train_raw), -CLIP_VAL, CLIP_VAL)
    X_val_xgb_s = np.clip(ae_scaler.transform(X_val_raw), -CLIP_VAL, CLIP_VAL)
    X_test_xgb_s = np.clip(ae_scaler.transform(X_test_raw), -CLIP_VAL, CLIP_VAL)

    z_train, err_train = get_ae_features(ae_model, X_train_xgb_s)
    z_val, err_val = get_ae_features(ae_model, X_val_xgb_s)
    z_test, err_test = get_ae_features(ae_model, X_test_xgb_s)

    X_train_aug = np.concatenate([X_train_xgb_s, z_train, err_train.reshape(-1, 1)], axis=1)
    X_val_aug = np.concatenate([X_val_xgb_s, z_val, err_val.reshape(-1, 1)], axis=1)
    X_test_aug = np.concatenate([X_test_xgb_s, z_test, err_test.reshape(-1, 1)], axis=1)

    scale_pos = float((y_train == 0).sum()) / float((y_train == 1).sum())

    # Monotone constraint: thu tu cot la [78 goc | 16 ae_latent | 1 ae_recon_error]
    # (khop chinh xac voi np.concatenate o tren). CHI ae_recon_error bi rang buoc
    # +1 (tai tao cang te => cang bat thuong). 16 ae_latent KHONG rang buoc vi
    # chung la toa do trong khong gian nen — gia tri cao/thap deu co the la benign,
    # khong co ngu nghia don dieu. 78 feature goc cung giu tu do (0).
    n_base = len(feature_cols)
    mono = build_monotone_constraints(
        n_base=n_base,
        n_augmented_free=BOTTLENECK_DIM,   # 16 latent: tu do
        n_augmented_monotone=1,            # 1 recon_error: don dieu tang
    )
    assert mono is None or len(mono) == X_train_aug.shape[1], (
        f"monotone_constraints do dai {len(mono)} != so cot {X_train_aug.shape[1]}"
    )

    if model_path.exists():
        model = joblib.load(model_path)
    else:
        model = xgb.XGBClassifier(
            **XGB_PARAMS,
            scale_pos_weight=scale_pos / XGB_SCALE_POS_DIVISOR,
            monotone_constraints=mono,
            device=XGB_DEVICE,
        )
        t0 = time.time()
        model.fit(X_train_aug, y_train, eval_set=[(X_val_aug, y_val)], verbose=False)
        print(f"    Training time: {time.time()-t0:.1f}s")
        joblib.dump(model, model_path)

    proba_train = model.predict_proba(X_train_aug)[:, 1]
    proba_val = model.predict_proba(X_val_aug)[:, 1]
    proba_test = model.predict_proba(X_test_aug)[:, 1]

    # Tune threshold theo F1-max tren TRAIN (khong phai val) — vi val_binary la
    # tap dung de BAO CAO known-attack F1/recall/FPR, tune tren chinh no se lam
    # threshold "biet truoc" tap eval, thoi phong ket qua. Threshold nay se duoc
    # gop trung binh qua 14 fold o main() roi dung LAI 1 threshold chung de
    # danh gia lai toan bo (xem main()).
    threshold, f1_train_opt, prec_train_opt, rec_train_opt = best_f1_threshold(y_train, proba_train)
    print(f"    [Train] best-F1 threshold={threshold:.4f}  F1={f1_train_opt:.4f}  "
          f"P={prec_train_opt:.4f}  R={rec_train_opt:.4f}")

    # SHAP top-15 (danh dau ro feature nao la AE-augmented)
    shap_path = images_dir / "shap_bar_top15.png"
    if not shap_path.exists():
        try:
            import shap
            augmented_feature_names = feature_cols + [f"ae_latent_{i}" for i in range(BOTTLENECK_DIM)] + ["ae_recon_error"]
            shap_idx = np.random.choice(len(X_val_aug), min(2000, len(X_val_aug)), replace=False)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val_aug[shap_idx])
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top15_idx = np.argsort(mean_abs_shap)[::-1][:15]
            top15_features = [augmented_feature_names[i] for i in top15_idx]
            colors = ["#e74c3c" if "ae_" in f else "#3498db" for f in top15_features]

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(range(len(top15_features)), mean_abs_shap[top15_idx][::-1], color=colors[::-1])
            ax.set_yticks(range(len(top15_features)))
            ax.set_yticklabels(top15_features[::-1], fontsize=9)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"AE+XGBoost SHAP Top 15 — held-out={label}\nRed = AE feature, Blue = original")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(shap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except ImportError:
            pass

    return {
        "threshold_from_train": threshold,
        "proba_val": proba_val,
        "y_val": y_val,
        "proba_test": proba_test,
        "y_test": y_test,
    }


def main() -> None:
    with open(LABELS_CONFIG_PATH, "r", encoding="utf-8") as f:
        labels_cfg = json.load(f)
    attack_labels = labels_cfg["attack_labels"]

    assert AE_SCALER_PATH.exists() and AE_MODEL_PATH.exists(), (
        f"Khong tim thay AE checkpoint tai {AE_ONLY_CKPT_DIR} — chay train_ae_only.py truoc."
    )
    print(f"Dang load AE checkpoint tu {AE_ONLY_CKPT_DIR} ...")
    ae_scaler = joblib.load(AE_SCALER_PATH)

    sample_df = load_split_csv(SPLITS_DIR / "by_label" / attack_labels[0] / "train_benign_attack.csv")
    feature_cols = get_feature_cols(sample_df)

    ae_model = BenignAutoencoder(input_dim=len(feature_cols), bottleneck_dim=BOTTLENECK_DIM).to(DEVICE)
    ae_model.load_state_dict(torch.load(AE_MODEL_PATH, map_location=DEVICE))
    ae_model.eval()
    print(f"XGBoost device: {XGB_DEVICE}  |  {len(feature_cols)} feature goc + {BOTTLENECK_DIM}+1 AE feature")

    results: dict = {"xgb_params": XGB_PARAMS,
                     "monotone_augmented": XGB_MONOTONE_AUGMENTED,
                     "monotone_n_constrained": 1 if XGB_MONOTONE_AUGMENTED else 0,
                     "monotone_note": "chi ae_recon_error bi rang buoc; 16 ae_latent tu do",
                     "scale_pos_divisor": XGB_SCALE_POS_DIVISOR,
                     "zero_shot_per_label": {}, "known_attack_per_label": {}}
    print(f"Monotone constraint: {'BAT' if XGB_MONOTONE_AUGMENTED else 'TAT'} "
          f"(chi ae_recon_error rang buoc don dieu; {BOTTLENECK_DIM} latent tu do)")

    # Phase 1: train tat ca 14 fold, chi thu threshold+proba tho, chua eval.
    raw_results: dict[str, dict] = {}
    for label in attack_labels:
        print(f"\n=== held-out = {label} ===")
        raw_results[label] = train_one_label(label, feature_cols, ae_model, ae_scaler)

    # Phase 2: tinh threshold trung binh tu 14 fold, danh gia lai CA 14 fold voi
    # CUNG 1 threshold nay (mo phong deployment thuc te chi co 1 nguong co dinh).
    threshold_per_label = {label: float(r["threshold_from_train"]) for label, r in raw_results.items()}
    threshold_mean = float(np.mean(list(threshold_per_label.values())))
    print(f"\n=== Threshold trung binh (tu 14 fold, tune tren train) = {threshold_mean:.4f} ===")
    print(f"    Threshold tung fold: { {k: round(v, 4) for k, v in threshold_per_label.items()} }")

    for label in attack_labels:
        r = raw_results[label]
        pred_val = (r["proba_val"] >= threshold_mean).astype(int)
        pred_test = (r["proba_test"] >= threshold_mean).astype(int)

        known_metrics = evaluate(r["y_val"], pred_val)
        zero_shot_metrics = evaluate(r["y_test"], pred_test)
        zero_shot_metrics["zero_acc"] = zero_shot_metrics["accuracy"]

        results["zero_shot_per_label"][label] = zero_shot_metrics
        results["known_attack_per_label"][label] = known_metrics
        print(f"  {label}: zero-shot recall={zero_shot_metrics['recall']:.4f} f1={zero_shot_metrics['f1']:.4f}  |  "
              f"known-attack recall={known_metrics['recall']:.4f} f1={known_metrics['f1']:.4f} "
              f"fpr={known_metrics['fpr']:.4f}")

    results["threshold_per_label_from_train"] = threshold_per_label
    results["threshold_mean_used_for_eval"] = threshold_mean

    def macro_avg(per_label: dict, key: str) -> float:
        vals = [m[key] for m in per_label.values() if m[key] is not None]
        return float(np.mean(vals)) if vals else None

    results["summary"] = {
        "zero_shot_macro_avg": {
            "f1": macro_avg(results["zero_shot_per_label"], "f1"),
            "recall": macro_avg(results["zero_shot_per_label"], "recall"),
            "precision": macro_avg(results["zero_shot_per_label"], "precision"),
            "zero_acc": macro_avg(results["zero_shot_per_label"], "zero_acc"),
            "fpr": macro_avg(results["zero_shot_per_label"], "fpr"),
        },
        "known_attack_macro_avg": {
            "f1": macro_avg(results["known_attack_per_label"], "f1"),
            "recall": macro_avg(results["known_attack_per_label"], "recall"),
            "precision": macro_avg(results["known_attack_per_label"], "precision"),
            "accuracy": macro_avg(results["known_attack_per_label"], "accuracy"),
            "fpr": macro_avg(results["known_attack_per_label"], "fpr"),
        },
    }
    print("\n=== SUMMARY (macro avg qua 14 held-out fold) ===")
    zs = results["summary"]["zero_shot_macro_avg"]
    ka = results["summary"]["known_attack_macro_avg"]
    print(f"  Zero-shot   : f1={zs['f1']:.4f}  recall={zs['recall']:.4f}  precision={zs['precision']:.4f}  "
          f"zero_acc={zs['zero_acc']:.4f}  fpr={zs['fpr']:.4f}")
    print(f"  Known-attack: f1={ka['f1']:.4f}  recall={ka['recall']:.4f}  precision={ka['precision']:.4f}  "
          f"accuracy={ka['accuracy']:.4f}  fpr={ka['fpr']:.4f}")

    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDa ghi metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
