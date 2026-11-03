from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler

import os

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from common.lgbm_config import LGBM_PARAMS, LGBM_SCALE_POS_DIVISOR, LGBM_EARLY_STOPPING_ROUNDS  # noqa: E402
from models.clad.util.metrics import best_f1_threshold, threshold_by_fpr  # noqa: E402

SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", BASE_DIR / "data" / "CICIDS2018"))
LABELS_CONFIG_PATH = Path(os.environ.get("LABELS_CONFIG_PATH", BASE_DIR / "config" / "labels.json"))

ARTIFACTS_ROOT = BASE_DIR / "artifacts" / "lgbm"
METRICS_PATH = ARTIFACTS_ROOT / "metrics.json"

SEED = 42
np.random.seed(SEED)


def load_split_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "Label" in df.columns and "Label_Text" in df.columns, f"{path}: missing Label/Label_Text"
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("Label", "Label_Text")]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


def evaluate_zero_shot(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float((y_true == y_pred).mean())
    return {"accuracy": acc, "n_samples": int(len(y_true))}


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
        model = lgb.LGBMClassifier(
            **LGBM_PARAMS,
            scale_pos_weight=scale_pos / LGBM_SCALE_POS_DIVISOR,
        )
        t0 = time.time()
        callbacks = [lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X_train_s,
                y_train,
                eval_set=[(X_val_s, y_val)],
                callbacks=callbacks,
            )
        print(f"    Training time: {time.time()-t0:.1f}s")
        joblib.dump(model, model_path)

    proba_train = model.predict_proba(X_train_s)[:, 1]
    proba_val = model.predict_proba(X_val_s)[:, 1]
    proba_test = model.predict_proba(X_test_s)[:, 1]

    TARGET_FPR = float(os.environ.get("TARGET_FPR", 0.005))
    proba_train_benign = proba_train[y_train == 0]
    threshold = threshold_by_fpr(proba_train_benign, target_fpr=TARGET_FPR)
    empirical_train_fpr = float((proba_train_benign >= threshold).mean())
    print(f"    [Train] Iso-FPR threshold={threshold:.4f}  (Train Benign FPR={empirical_train_fpr*100:.3f}% "
          f"on {len(proba_train_benign):,} samples, target={TARGET_FPR*100:.1f}%)")

    shap_path = images_dir / "shap_bar_top15.png"
    if not shap_path.exists():
        try:
            import shap
            shap_idx = np.random.choice(len(X_val_s), min(2000, len(X_val_s)), replace=False)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val_s[shap_idx])
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top15_idx = np.argsort(mean_abs_shap)[::-1][:15]
            top15_features = [feature_cols[i] for i in top15_idx]

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(range(len(top15_features)), mean_abs_shap[top15_idx][::-1], color="#2ecc71")
            ax.set_yticks(range(len(top15_features)))
            ax.set_yticklabels(top15_features[::-1], fontsize=9)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"LightGBM-only SHAP Top 15 — held-out={label}")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(shap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception:
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

    sample_df = load_split_csv(SPLITS_DIR / "by_label" / attack_labels[0] / "train_benign_attack.csv")
    feature_cols = get_feature_cols(sample_df)
    print(f"LightGBM model  |  {len(feature_cols)} features  |  {len(attack_labels)} folds")

    results: dict = {"lgbm_params": LGBM_PARAMS, "zero_shot_per_label": {}, "known_attack_per_label": {}}

    raw_results: dict[str, dict] = {}
    for label in attack_labels:
        print(f"\n=== held-out = {label} ===")
        raw_results[label] = train_one_label(label, feature_cols)

    threshold_per_label = {label: float(r["threshold_from_train"]) for label, r in raw_results.items()}
    threshold_mean = float(np.mean(list(threshold_per_label.values())))
    print(f"\n=== Threshold trung binh (tu 14 fold, tune tren train) = {threshold_mean:.4f} ===")
    print(f"    Threshold tung fold: { {k: round(v, 4) for k, v in threshold_per_label.items()} }")

    for label in attack_labels:
        r = raw_results[label]
        pred_val = (r["proba_val"] >= threshold_mean).astype(int)
        pred_test = (r["proba_test"] >= threshold_mean).astype(int)

        known_metrics = evaluate(r["y_val"], pred_val)
        zero_shot_metrics = evaluate_zero_shot(r["y_test"], pred_test)

        results["zero_shot_per_label"][label] = zero_shot_metrics
        results["known_attack_per_label"][label] = known_metrics
        print(f"  {label}: zero-shot accuracy={zero_shot_metrics['accuracy']:.4f}  |  "
              f"known-attack recall={known_metrics['recall']:.4f} f1={known_metrics['f1']:.4f} "
              f"fpr={known_metrics['fpr']:.4f}")

    results["threshold_per_label_from_train"] = threshold_per_label
    results["threshold_mean_used_for_eval"] = threshold_mean

    def macro_avg(per_label: dict, key: str) -> float:
        vals = [m[key] for m in per_label.values() if m[key] is not None]
        return float(np.mean(vals)) if vals else None

    results["summary"] = {
        "zero_shot_macro_avg": {
            "accuracy": macro_avg(results["zero_shot_per_label"], "accuracy"),
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
    print(f"  Zero-shot   : accuracy={zs['accuracy']:.4f}")
    print(f"  Known-attack: f1={ka['f1']:.4f}  recall={ka['recall']:.4f}  precision={ka['precision']:.4f}  "
          f"accuracy={ka['accuracy']:.4f}  fpr={ka['fpr']:.4f}")

    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDa ghi metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
