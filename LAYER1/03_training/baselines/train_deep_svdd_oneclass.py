"""Train and evaluate single-sphere One-Class Deep SVDD on CICIDS2018.

This is a tabular-data adaptation of the One-Class Deep SVDD objective in
Ruff et al. (ICML 2018): train on benign rows only, keep one fixed center, and
score each sample by squared distance to that center. It intentionally does
not use K-Means or multiple spheres.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler

from models.deep_svdd_oneclass import OneClassDeepSVDD


BASE_DIR = Path(__file__).resolve().parent
SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", BASE_DIR / "data" / "CICIDS2018"))
LABELS_CONFIG_PATH = Path(os.environ.get("LABELS_CONFIG_PATH", BASE_DIR / "config" / "labels.json"))
# Separate output keeps the previous 12-sphere Mixture Deep SVDD artifacts intact.
ARTIFACTS_DIR = Path(os.environ.get(
    "DEEP_SVDD_ARTIFACTS_ROOT", BASE_DIR / "artifacts" / "deep_svdd_paper"
))
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoint"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"
MODEL_PATH = CHECKPOINT_DIR / "oneclass_deep_svdd.pt"
TRAINING_STATE_PATH = CHECKPOINT_DIR / "training_state.pt"

SEED = 42
EMBED_DIM = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
EPOCHS = 100
BATCH_SIZE = 2048
FPR_BUDGET = 0.005
CLIP_VAL = 20.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path: Path, expected_features: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    df = pd.read_csv(path)
    if "Label" not in df or "Label_Text" not in df:
        raise ValueError(f"{path}: expected Label and Label_Text columns")
    features = [c for c in df.columns if c not in ("Label", "Label_Text")]
    if expected_features is not None and features != expected_features:
        raise ValueError(f"{path}: feature columns/order differ from benign training data")
    return df


def benign_quantile_threshold(scores: np.ndarray) -> float:
    """Calibrate one common cutoff with empirical benign FPR no greater than 0.5%."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected a nonempty 1-D array of finite benign scores")
    max_false_positives = int(np.floor(len(scores) * FPR_BUDGET))
    boundary_index = len(scores) - max_false_positives - 1
    empirical_boundary = np.partition(scores, boundary_index)[boundary_index]
    nominal_quantile = np.quantile(scores, 1.0 - FPR_BUDGET)
    threshold = float(max(empirical_boundary, nominal_quantile))
    assert np.count_nonzero(scores > threshold) <= max_false_positives
    return threshold


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "n_samples": int(len(y_true)),
    }


def macro_avg(rows: dict, key: str) -> float | None:
    values = [row[key] for row in rows.values() if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def main() -> None:
    import torch

    seed_everything(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. In Colab choose Runtime > Change runtime type > GPU, "
            "then rerun this script."
        )
    device = "cuda"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    labels_cfg = json.loads(LABELS_CONFIG_PATH.read_text(encoding="utf-8"))
    attack_labels = labels_cfg["attack_labels"]
    train_df = load_split(SPLITS_DIR / "train_benign_only.csv")
    feature_cols = [c for c in train_df.columns if c not in ("Label", "Label_Text")]
    if not np.all(train_df["Label"].to_numpy() == 0):
        raise ValueError("train_benign_only.csv must contain benign rows only")
    X_train_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
    print(f"One-Class Deep SVDD | {device} | {len(X_train_raw):,} benign rows | "
          f"{len(feature_cols)} features | 1 sphere", flush=True)

    scaler = RobustScaler()
    X_train = np.clip(scaler.fit_transform(X_train_raw), -CLIP_VAL, CLIP_VAL).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)
    print(f"RobustScaler fit on benign-only training data: {SCALER_PATH}", flush=True)

    model = OneClassDeepSVDD(
        input_dim=len(feature_cols), embed_dim=EMBED_DIM,
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS, batch_size=BATCH_SIZE, device=device,
    )
    resume = os.environ.get("DEEP_SVDD_RESUME", "1").strip().lower() not in {"0", "false", "no"}
    t0 = time.time()
    model.fit(X_train, str(TRAINING_STATE_PATH), resume=resume)
    print(f"Training/resume elapsed: {time.time() - t0:.1f}s", flush=True)
    model.save(str(MODEL_PATH))

    print("Calibrating the shared threshold on benign training scores ...", flush=True)
    benign_scores = model.decision_scores(X_train)
    threshold = benign_quantile_threshold(benign_scores)
    calibration_fpr = float(np.mean(benign_scores > threshold))
    print(f"  q={1.0 - FPR_BUDGET:.3f} threshold={threshold:.8g} "
          f"benign-train FPR={calibration_fpr:.4%} n={len(benign_scores):,}", flush=True)

    results = {
        "model": "OneClassDeepSVDD_Ruff2018_style",
        "paper": "Ruff et al., Deep One-Class Classification, ICML 2018",
        "implementation_note": (
            "Single fixed-center hypersphere and one-class objective; MLP adapted to "
            "78-dimensional CICIDS tabular features. No K-Means/multiple spheres. "
            "DCAE pretraining from the image experiments is not used."
        ),
        "training_data": "train_benign_only.csv",
        "parameters": {
            "architecture": [len(feature_cols), 256, 128, EMBED_DIM],
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay_lambda": WEIGHT_DECAY,
            "objective": "mean squared embedding distance to fixed center + L2 weight decay",
            "seed": SEED,
        },
        "threshold_rule": (
            "shared 99.5th percentile of benign training anomaly scores; empirical "
            "tie-safe guard ensures FPR <= 0.005; predict attack if score > threshold"
        ),
        "target_calibration_fpr": FPR_BUDGET,
        "threshold": threshold,
        "benign_train_fpr": calibration_fpr,
        "benign_train_n": int(len(benign_scores)),
        "zero_shot_per_label": {},
        "known_attack_per_label": {},
    }

    print("Evaluating zero-shot attack sets and known-attack validation folds ...", flush=True)
    for label in attack_labels:
        test_df = load_split(SPLITS_DIR / "by_label" / label / "test_zeroshot.csv", feature_cols)
        X_test = np.clip(scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32)),
                         -CLIP_VAL, CLIP_VAL).astype(np.float32)
        test_scores = model.decision_scores(X_test)
        test_pred = (test_scores > threshold).astype(np.int64)
        test_y = test_df["Label"].to_numpy(dtype=np.int64)
        results["zero_shot_per_label"][label] = {
            "accuracy": float(np.mean(test_y == test_pred)),
            "n_samples": int(len(test_y)),
        }

        val_df = load_split(SPLITS_DIR / "by_label" / label / "val_binary.csv", feature_cols)
        X_val = np.clip(scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float32)),
                        -CLIP_VAL, CLIP_VAL).astype(np.float32)
        val_scores = model.decision_scores(X_val)
        val_y = val_df["Label"].to_numpy(dtype=np.int64)
        results["known_attack_per_label"][label] = evaluate(val_y, val_scores, threshold)
        known = results["known_attack_per_label"][label]
        print(f"  {label:30s} zero-shot accuracy={results['zero_shot_per_label'][label]['accuracy']:.4f} | "
              f"known F1={known['f1']:.4f} recall={known['recall']:.4f} "
              f"FPR={known['fpr']:.4f}", flush=True)

    results["summary"] = {
        "zero_shot_macro_avg": {
            "accuracy": macro_avg(results["zero_shot_per_label"], "accuracy")
        },
        "known_attack_macro_avg": {
            key: macro_avg(results["known_attack_per_label"], key)
            for key in ("f1", "recall", "precision", "accuracy", "fpr")
        },
    }
    METRICS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(results["summary"], indent=2))
    print(f"Metrics: {METRICS_PATH}")
    print(f"Model/scaler/checkpoints: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
