"""
Train XGBoost-only baseline — train RIENG 14 lan, moi lan 1 held-out label,
tren train_benign_attack.csv/val_binary.csv cua chinh label do (KHONG augment
feature — chi 78 feature CICFlowMeter goc). Day la baseline "khong co AE/SVDD"
de so sanh voi 2 bien the augmented (AE+XGBoost, SVDD+XGBoost) SAU NAY —
ca 3 bien the DUNG CHUNG 1 config XGBoost (common/xgb_config.py) de so sanh cong bang.

Pipeline (moi held-out label):
  1. Doc train_benign_attack.csv (Benign + 13 attack known) + val_binary.csv cua label do.
  2. RobustScaler fit tren train_benign_attack (KHONG fit rieng tren benign-only —
     khac AE/SVDD, vi XGBoost train tren ca benign+attack ngay tu dau). KHONG clip
     sau scale — da thu nghiem them np.clip(-20,20) va xac nhan lam ZS-recall macro
     GIAM (0.6761 -> 0.5778, xem RESULTS.md) vi mot so cot CICFlowMeter co outlier
     cuc doan (Flow/Fwd IAT Min/Max, Idle Max/Mean/Std) LAI CHINH LA tin hieu phan
     biet that su cho vai loai tan cong (vd DDOS_attack_LOIC_UDP giam tu 1.0 xuong
     0.18 sau khi clip) — khong phai nhieu can loai bo. XGBoost-only khong dung
     chung scaler voi bat ky encoder nao khac nen khong gap loi kien truc 2-scaler
     nhu AE+XGBoost/SVDD+XGBoost.
  3. Train XGBClassifier (config trong common/xgb_config.py), early stopping tren val.
  4. Tune threshold F1-max tren TRAIN (train_benign_attack.csv), khong phai val —
     vi val_binary.csv la tap DUNG DE BAO CAO known-attack F1/recall/FPR; tune
     threshold tren chinh tap eval se lam ket qua bi thoi phong (data leak kieu
     "threshold biet truoc dap an"). Sau khi train xong CA 14 fold, lay TRUNG
     BINH 14 threshold nay thanh 1 threshold duy nhat, roi dung LAI threshold
     chung do de eval lai toan bo 14 fold — mo phong dung mot nguong quyet dinh
     co dinh nhu khi trien khai thuc te (thay vi 14 threshold khac nhau).
  5. Eval ZERO-SHOT tren test_zeroshot.csv cua chinh label do (attack hoan toan unseen),
     dung threshold TRUNG BINH chung (khong tune rieng tren test_zeroshot — tranh leak).
  6. Eval KNOWN-ATTACK tren val_binary.csv (Benign + 13 attack known) — FPR/Precision
     co y nghia that vi co Benign that trong tap nay.
  7. Luu checkpoint (model.pkl + scaler.pkl) + SHAP bar chart top-15 feature vao
     artifacts/xgboost_only/<label>/.

Sau 14 lan: ghi metrics.json tong hop (zero_shot_per_label, known_attack_per_label,
summary macro avg qua 14 fold) — giong cau truc AE-Only/SVDD-Only de de so sanh.
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
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.xgb_config import XGB_PARAMS, XGB_SCALE_POS_DIVISOR  # noqa: E402
from models.clad.util.metrics import best_f1_threshold  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = BASE_DIR / "03_splits_v2"
LABELS_CONFIG_PATH = BASE_DIR / "config" / "labels.json"

ARTIFACTS_ROOT = Path(__file__).resolve().parent / "artifacts" / "xgboost_only"
METRICS_PATH = ARTIFACTS_ROOT / "metrics.json"

SEED = 42
np.random.seed(SEED)

try:
    import torch
    XGB_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    XGB_DEVICE = "cpu"


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


def train_one_label(label: str, feature_cols: list[str]) -> dict:
    label_artifacts_dir = ARTIFACTS_ROOT / label
    images_dir = label_artifacts_dir / "images"
    ckpt_dir = label_artifacts_dir / "checkpoint"
    images_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = ckpt_dir / "scaler.pkl"
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

    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        scaler = RobustScaler()
        scaler.fit(X_train_raw)
        joblib.dump(scaler, scaler_path)

    X_train_s = scaler.transform(X_train_raw)
    X_val_s = scaler.transform(X_val_raw)
    X_test_s = scaler.transform(X_test_raw)

    scale_pos = float((y_train == 0).sum()) / float((y_train == 1).sum())

    if model_path.exists():
        model = joblib.load(model_path)
    else:
        model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos / XGB_SCALE_POS_DIVISOR, device=XGB_DEVICE)
        t0 = time.time()
        model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
        print(f"    Training time: {time.time()-t0:.1f}s")
        joblib.dump(model, model_path)

    proba_train = model.predict_proba(X_train_s)[:, 1]
    proba_val = model.predict_proba(X_val_s)[:, 1]
    proba_test = model.predict_proba(X_test_s)[:, 1]

    # Tune threshold theo F1-max tren TRAIN (khong phai val) — vi val_binary la
    # tap dung de BAO CAO known-attack F1/recall/FPR, tune tren chinh no se lam
    # threshold "biet truoc" tap eval, thoi phong ket qua. Threshold nay se duoc
    # gop trung binh qua 14 fold o main() roi dung LAI 1 threshold chung de
    # danh gia lai toan bo (xem main()).
    threshold, f1_train_opt, prec_train_opt, rec_train_opt = best_f1_threshold(y_train, proba_train)
    print(f"    [Train] best-F1 threshold={threshold:.4f}  F1={f1_train_opt:.4f}  "
          f"P={prec_train_opt:.4f}  R={rec_train_opt:.4f}")

    # SHAP top-15 feature bar chart (chi neu chua co, tranh tinh lai moi lan re-run)
    shap_path = images_dir / "shap_bar_top15.png"
    if not shap_path.exists():
        try:
            import shap
            shap_idx = np.random.choice(len(X_val_s), min(2000, len(X_val_s)), replace=False)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val_s[shap_idx])
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top15_idx = np.argsort(mean_abs_shap)[::-1][:15]
            top15_features = [feature_cols[i] for i in top15_idx]

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(range(len(top15_features)), mean_abs_shap[top15_idx][::-1], color="#3498db")
            ax.set_yticks(range(len(top15_features)))
            ax.set_yticklabels(top15_features[::-1], fontsize=9)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"XGBoost-only SHAP Top 15 — held-out={label}")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(shap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except ImportError:
            pass  # shap khong bat buoc, bo qua neu chua cai

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

    sample_df = load_split_csv(SPLITS_DIR / "by_label" / attack_labels[0] / "train_benign_attack.csv")
    feature_cols = get_feature_cols(sample_df)
    print(f"XGBoost device: {XGB_DEVICE}  |  {len(feature_cols)} feature")

    results: dict = {"xgb_params": XGB_PARAMS, "zero_shot_per_label": {}, "known_attack_per_label": {}}

    # Phase 1: train tat ca 14 fold, chi thu threshold+proba tho, chua eval.
    raw_results: dict[str, dict] = {}
    for label in attack_labels:
        print(f"\n=== held-out = {label} ===")
        raw_results[label] = train_one_label(label, feature_cols)

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
