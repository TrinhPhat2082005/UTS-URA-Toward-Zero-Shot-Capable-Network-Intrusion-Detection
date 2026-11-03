"""Pure linear SVM baseline on the 78 original CICFlowMeter features.

Each leave-one-attack-out fold has its own model and its own threshold. The
threshold is calibrated from that fold's 560k benign training scores at the
99.5th percentile, without using attack labels or validation/test scores.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", BASE_DIR / "data" / "CICIDS2018"))
LABELS_CONFIG_PATH = Path(os.environ.get("LABELS_CONFIG_PATH", BASE_DIR / "config" / "labels.json"))
ARTIFACTS_ROOT = Path(os.environ.get("SVM_ARTIFACTS_ROOT", BASE_DIR / "artifacts" / "svm_only"))
METRICS_PATH = ARTIFACTS_ROOT / "metrics.json"

SEED = 42
SVM_C = 1.0
MAX_ITER = 5000
FPR_BUDGET = 0.005


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in ("Label", "Label_Text")]


def load_split(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Label" not in df or "Label_Text" not in df:
        raise ValueError(f"{path}: missing Label or Label_Text")
    if columns is not None and feature_columns(df) != columns:
        raise ValueError(f"{path}: feature columns do not match training fold")
    return df


def benign_quantile_threshold(scores: np.ndarray, fpr_budget: float = FPR_BUDGET) -> float:
    """Return a strict-'>' cutoff with empirical benign FPR <= fpr_budget.

    The 99.5th percentile is the nominal cutoff. With tied scores or a small
    sample, the order-statistic bound prevents exceeding the FPR budget.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected a nonempty 1-D array of finite benign scores")
    if not 0 <= fpr_budget < 1:
        raise ValueError("fpr_budget must be in [0, 1)")
    max_false_positives = int(np.floor(len(scores) * fpr_budget))
    boundary_index = len(scores) - max_false_positives - 1
    boundary = np.partition(scores, boundary_index)[boundary_index]
    quantile = np.quantile(scores, 1.0 - fpr_budget)
    threshold = float(max(boundary, quantile))
    assert np.count_nonzero(scores > threshold) <= max_false_positives
    return threshold


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "accuracy": float((tp + tn) / len(y_true)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "n_samples": int(len(y_true)),
    }


def train_one_fold(label: str, columns: list[str]) -> tuple[float, dict, dict, float]:
    fold_dir = SPLITS_DIR / "by_label" / label
    train_df = load_split(fold_dir / "train_benign_attack.csv", columns)
    val_df = load_split(fold_dir / "val_binary.csv", columns)
    test_df = load_split(fold_dir / "test_zeroshot.csv", columns)

    x_train = train_df[columns].to_numpy(dtype=np.float32)
    y_train = train_df["Label"].to_numpy(dtype=np.int64)
    x_val = val_df[columns].to_numpy(dtype=np.float32)
    y_val = val_df["Label"].to_numpy(dtype=np.int64)
    x_test = test_df[columns].to_numpy(dtype=np.float32)
    y_test = test_df["Label"].to_numpy(dtype=np.int64)
    if not np.isin(y_train, [0, 1]).all() or not np.isin(y_val, [0, 1]).all():
        raise ValueError(f"{label}: expected binary Label values 0/1")
    if not np.all(y_test == 1) or not np.all(test_df["Label_Text"] == label):
        raise ValueError(f"{label}: zero-shot test contains unexpected labels")

    checkpoint_dir = ARTIFACTS_ROOT / label / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "model.pkl"
    if model_path.exists():
        print(f"  [Checkpoint] Loading {model_path}", flush=True)
        model = joblib.load(model_path)
    else:
        model = Pipeline([
            ("scale", RobustScaler()),
            ("svm", LinearSVC(C=SVM_C, class_weight="balanced", max_iter=MAX_ITER,
                              dual=False, random_state=SEED)),
        ])
        start = time.time()
        model.fit(x_train, y_train)
        joblib.dump(model, model_path)
        print(f"  Training time: {time.time() - start:.1f}s", flush=True)

    # LinearSVC returns a signed margin, not a probability. Higher = attack.
    benign_scores = model.decision_function(x_train[y_train == 0])
    threshold = benign_quantile_threshold(benign_scores)
    calibration_fpr = float(np.mean(benign_scores > threshold))
    known = evaluate(y_val, model.decision_function(x_val), threshold)
    zero_shot = evaluate(y_test, model.decision_function(x_test), threshold)
    print(f"  threshold={threshold:.6g}  benign-train FPR={calibration_fpr:.4%}  "
          f"val FPR={known['fpr']:.4%}  val F1={known['f1']:.4f}  "
          f"zero-shot accuracy={zero_shot['accuracy']:.4f}", flush=True)
    return threshold, known, {
        "accuracy": zero_shot["accuracy"], "n_samples": zero_shot["n_samples"]
    }, calibration_fpr


def main() -> None:
    attack_labels = json.loads(LABELS_CONFIG_PATH.read_text(encoding="utf-8"))["attack_labels"]
    sample = load_split(SPLITS_DIR / "by_label" / attack_labels[0] / "train_benign_attack.csv")
    columns = feature_columns(sample)
    print(f"Pure LinearSVC | {len(columns)} original features | {len(attack_labels)} LOAO folds", flush=True)
    del sample

    results = {
        "model": "RobustScaler+LinearSVC",
        "parameters": {"svm_C": SVM_C, "class_weight": "balanced", "max_iter": MAX_ITER},
        "threshold_rule": "per-fold 99.5th percentile of benign training margins; predict attack if score > threshold",
        "target_calibration_fpr": FPR_BUDGET,
        "threshold_per_label": {}, "benign_train_fpr_per_label": {},
        "zero_shot_per_label": {}, "known_attack_per_label": {},
    }
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    for label in attack_labels:
        print(f"\n=== held-out = {label} ===", flush=True)
        threshold, known, zero_shot, calibration_fpr = train_one_fold(label, columns)
        results["threshold_per_label"][label] = threshold
        results["benign_train_fpr_per_label"][label] = calibration_fpr
        results["known_attack_per_label"][label] = known
        results["zero_shot_per_label"][label] = zero_shot
        (ARTIFACTS_ROOT / "metrics_partial.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    results["summary"] = {
        "zero_shot_macro_avg": {
            "accuracy": float(np.mean([row["accuracy"] for row in results["zero_shot_per_label"].values()]))
        },
        "known_attack_macro_avg": {
            key: float(np.mean([row[key] for row in results["known_attack_per_label"].values()]))
            for key in ("f1", "recall", "precision", "accuracy", "fpr")
        },
    }
    METRICS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved metrics: {METRICS_PATH}", flush=True)


if __name__ == "__main__":
    main()
