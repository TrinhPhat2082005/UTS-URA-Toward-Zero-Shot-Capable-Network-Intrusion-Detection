from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

BASE_DIR = Path(__file__).resolve().parent
SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", BASE_DIR / "data" / "CICIDS2018"))
LABELS_CONFIG_PATH = Path(os.environ.get("LABELS_CONFIG_PATH", BASE_DIR / "config" / "labels.json"))
ARTIFACTS_ROOT = Path(os.environ.get("RF_ARTIFACTS_ROOT", BASE_DIR / "artifacts" / "random_forest"))
METRICS_PATH = ARTIFACTS_ROOT / "metrics.json"

SEED = 42
N_ESTIMATORS = 200
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 2
FPR_BUDGET = 0.005
np.random.seed(SEED)


def load_split_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Label" not in df.columns or "Label_Text" not in df.columns:
        raise ValueError(f"{path}: missing Label/Label_Text columns")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("Label", "Label_Text")]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "n_samples": int(len(y_true)),
    }


def benign_quantile_threshold(scores: np.ndarray) -> float:
    """Calibrate an attack-probability cutoff with benign FPR <= 0.5%.

    The nominal cutoff is quantile(0.995). The order-statistic guard handles
    tied probabilities conservatively; inference classifies attack only when
    score > threshold.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected a nonempty 1-D array of finite benign scores")
    max_false_positives = int(np.floor(len(scores) * FPR_BUDGET))
    boundary_index = len(scores) - max_false_positives - 1
    boundary = np.partition(scores, boundary_index)[boundary_index]
    return float(max(boundary, np.quantile(scores, 1.0 - FPR_BUDGET)))


def train_one_label(label: str, feature_cols: list[str]) -> dict:
    fold_dir = ARTIFACTS_ROOT / label
    ckpt_dir = fold_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = ckpt_dir / "model_sklearn_cpu.pkl"

    train_df = load_split_csv(SPLITS_DIR / "by_label" / label / "train_benign_attack.csv")
    val_df = load_split_csv(SPLITS_DIR / "by_label" / label / "val_binary.csv")
    test_df = load_split_csv(SPLITS_DIR / "by_label" / label / "test_zeroshot.csv")
    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["Label"].to_numpy(dtype=np.int64)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df["Label"].to_numpy(dtype=np.int64)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["Label"].to_numpy(dtype=np.int64)

    if model_path.exists():
        print(f"    [Checkpoint] Loading {model_path}")
        model = joblib.load(model_path)
    else:
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features="sqrt",
            class_weight="balanced_subsample",
            bootstrap=True,
            n_jobs=-1,
            random_state=SEED,
        )
        t0 = time.time()
        model.fit(X_train, y_train)
        print(f"    Training time: {time.time() - t0:.1f}s")
        joblib.dump(model, model_path)

    positive_col = int(np.flatnonzero(model.classes_ == 1)[0])
    train_scores = model.predict_proba(X_train)[:, positive_col]
    val_scores = model.predict_proba(X_val)[:, positive_col]
    test_scores = model.predict_proba(X_test)[:, positive_col]
    benign_train_scores = train_scores[y_train == 0]
    threshold = benign_quantile_threshold(benign_train_scores)
    calibration_fpr = float(np.mean(benign_train_scores > threshold))
    print(f"    [Calibration] q=0.995 threshold={threshold:.6f}  "
          f"benign-train FPR={calibration_fpr:.4%}")
    return {"threshold_from_benign_train": threshold,
            "benign_train_fpr": calibration_fpr, "y_val": y_val,
            "scores_val": val_scores, "y_test": y_test, "scores_test": test_scores}


def macro_avg(rows: dict, key: str) -> float | None:
    values = [row[key] for row in rows.values() if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def main() -> None:
    labels_cfg = json.loads(LABELS_CONFIG_PATH.read_text(encoding="utf-8"))
    attack_labels = labels_cfg["attack_labels"]
    sample_df = load_split_csv(SPLITS_DIR / "by_label" / attack_labels[0] / "train_benign_attack.csv")
    feature_cols = get_feature_cols(sample_df)
    print(f"Random Forest | {len(feature_cols)} features | {N_ESTIMATORS} trees | scikit-learn CPU (multi-core)")

    raw = {}
    for label in attack_labels:
        print(f"\n=== held-out = {label} ===")
        raw[label] = train_one_label(label, feature_cols)

    thresholds = {label: row["threshold_from_benign_train"] for label, row in raw.items()}
    calibration_fprs = {label: row["benign_train_fpr"] for label, row in raw.items()}
    results = {
        "model": "RandomForestClassifier",
        "backend": "scikit-learn CPU (n_jobs=-1)",
        "parameters": {"n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH,
                       "min_samples_leaf": MIN_SAMPLES_LEAF, "max_features": "sqrt",
                       "bootstrap": True, "class_weight": "balanced_subsample"},
        "threshold_rule": "per-fold 99.5th percentile of benign training probabilities; predict attack if score > threshold",
        "target_calibration_fpr": FPR_BUDGET,
        "threshold_per_label_from_benign_train": thresholds,
        "benign_train_fpr_per_label": calibration_fprs,
        "zero_shot_per_label": {},
        "known_attack_per_label": {},
    }
    for label, row in raw.items():
        threshold = thresholds[label]
        val_pred = (row["scores_val"] > threshold).astype(np.int64)
        test_pred = (row["scores_test"] > threshold).astype(np.int64)
        known = evaluate(row["y_val"], val_pred)
        zero_shot = {"accuracy": float((row["y_test"] == test_pred).mean()),
                     "n_samples": int(len(row["y_test"]))}
        results["known_attack_per_label"][label] = known
        results["zero_shot_per_label"][label] = zero_shot
        print(f"  {label}: zero-shot accuracy={zero_shot['accuracy']:.4f} | "
              f"known-attack F1={known['f1']:.4f} recall={known['recall']:.4f} fpr={known['fpr']:.4f}")

    results["summary"] = {
        "zero_shot_macro_avg": {"accuracy": macro_avg(results["zero_shot_per_label"], "accuracy")},
        "known_attack_macro_avg": {key: macro_avg(results["known_attack_per_label"], key)
                                   for key in ("f1", "recall", "precision", "accuracy", "fpr")},
    }
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDa ghi metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
