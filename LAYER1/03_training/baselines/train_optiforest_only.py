from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "OptIForest"))

from common.optiforest_config import OPTIFOREST_PARAMS
from detectors import OptIForest
from models.clad.util.metrics import threshold_by_fpr

SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", BASE_DIR / "data" / "CICIDS2018"))
LABELS_CONFIG_PATH = Path(os.environ.get("LABELS_CONFIG_PATH", BASE_DIR / "config" / "labels.json"))

ARTIFACTS_DIR = BASE_DIR / "artifacts" / "optiforest"
IMAGES_DIR = ARTIFACTS_DIR / "images"
CKPT_DIR = ARTIFACTS_DIR / "checkpoint"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

for d in (IMAGES_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SCALER_PATH = CKPT_DIR / "scaler.pkl"
MODEL_PATH = CKPT_DIR / "optiforest_model.pkl"
BENIGN_CACHE_PATH = CKPT_DIR / "val_benign_scores.npy"
TRAIN_BENIGN_CACHE_PATH = CKPT_DIR / "train_benign_scores.npy"

SEED = 42
np.random.seed(SEED)
CLIP_VAL = 20.0


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


def predict_anomaly_scores(
    model: OptIForest, X: np.ndarray, chunk_size: int = 2500, n_jobs: int = -1
) -> np.ndarray:
    """
    OptIForest.decision_function(X) returns -1.0 * avg_depths (where shorter path = anomaly).
    Multiplying by -1.0 converts it into positive anomaly score: higher = more abnormal.
    Inference is parallelized in chunks across available CPU cores.
    """
    N = len(X)
    if N == 0:
        return np.array([], dtype=np.float32)
    if N <= chunk_size:
        return -1.0 * model.decision_function(X)

    chunks = [X[i : i + chunk_size] for i in range(0, N, chunk_size)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(model.decision_function)(chunk) for chunk in chunks
    )
    return -1.0 * np.concatenate(results)


def main() -> None:
    with open(LABELS_CONFIG_PATH, "r", encoding="utf-8") as f:
        labels_cfg = json.load(f)
    attack_labels = labels_cfg["attack_labels"]

    print(f"=== OptIForest Baseline (Xiang et al., 2023) ===")
    print(f"Params: {OPTIFOREST_PARAMS}")

    benign_train_path = SPLITS_DIR / "train_benign_only.csv"
    print(f"Loading {benign_train_path} ...")
    benign_df = load_split_csv(benign_train_path)
    feature_cols = get_feature_cols(benign_df)

    X_train_raw = benign_df[feature_cols].values.astype("float32")

    if SCALER_PATH.exists():
        print(f"[Checkpoint] Loading scaler from {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
    else:
        print(f"Fitting RobustScaler on {len(X_train_raw):,} benign samples ...")
        scaler = RobustScaler()
        scaler.fit(X_train_raw)
        joblib.dump(scaler, SCALER_PATH)

    X_train_s = np.clip(scaler.transform(X_train_raw), -CLIP_VAL, CLIP_VAL)

    if MODEL_PATH.exists():
        print(f"[Checkpoint] Found {MODEL_PATH} - loading, skip training.")
        model = joblib.load(MODEL_PATH)
    else:
        print(f"Training OptIForest ({OPTIFOREST_PARAMS['num_trees']} trees, cut_threshold={OPTIFOREST_PARAMS['threshold']}) ...")
        model = OptIForest(
            lsh_family=OPTIFOREST_PARAMS["lsh_family"],
            num_trees=OPTIFOREST_PARAMS["num_trees"],
            threshold=OPTIFOREST_PARAMS["threshold"],
            branch=OPTIFOREST_PARAMS["branch"],
            granularity=OPTIFOREST_PARAMS["granularity"],
        )
        t0 = time.time()
        model.fit(X_train_s)
        print(f"OptIForest fit completed in {time.time()-t0:.1f}s")
        joblib.dump(model, MODEL_PATH)

    TARGET_FPR = float(os.environ.get("TARGET_FPR", 0.005))
    if TRAIN_BENIGN_CACHE_PATH.exists():
        print(f"[Cache] Loading cached Train Benign scores from {TRAIN_BENIGN_CACHE_PATH}")
        scores_train_benign = np.load(TRAIN_BENIGN_CACHE_PATH)
    else:
        print(f"Scoring {len(X_train_s):,} Train Benign samples ...")
        t0 = time.time()
        scores_train_benign = predict_anomaly_scores(model, X_train_s)
        print(f"Train Benign scoring completed in {time.time()-t0:.1f}s")
        np.save(TRAIN_BENIGN_CACHE_PATH, scores_train_benign)

    opt_threshold = threshold_by_fpr(scores_train_benign, target_fpr=TARGET_FPR)
    empirical_train_fpr = float((scores_train_benign >= opt_threshold).mean())
    print(f"  OptIForest threshold (Quantile {1.0-TARGET_FPR:.3f}): {opt_threshold:.6f}  "
          f"(Train Benign FPR={empirical_train_fpr*100:.3f}%)")

    threshold_label = attack_labels[0]
    val_ref_df = load_split_csv(SPLITS_DIR / "by_label" / threshold_label / "val_binary.csv")
    X_val_ref_raw = val_ref_df[feature_cols].values.astype("float32")
    X_val_ref_s = np.clip(scaler.transform(X_val_ref_raw), -CLIP_VAL, CLIP_VAL)
    y_val_ref = val_ref_df["Label"].values

    benign_mask_ref = (y_val_ref == 0)
    attack_mask_ref = ~benign_mask_ref

    if BENIGN_CACHE_PATH.exists():
        print(f"[Cache] Loading cached Benign validation scores from {BENIGN_CACHE_PATH}")
        cached_benign_scores = np.load(BENIGN_CACHE_PATH)
    else:
        print(f"Scoring {benign_mask_ref.sum():,} Benign validation samples (done once, cached for all 14 folds)...")
        t0 = time.time()
        cached_benign_scores = predict_anomaly_scores(model, X_val_ref_s[benign_mask_ref])
        print(f"Benign scoring completed in {time.time()-t0:.1f}s")
        np.save(BENIGN_CACHE_PATH, cached_benign_scores)

    scores_val_ref = np.empty(len(y_val_ref), dtype=np.float32)
    scores_val_ref[benign_mask_ref] = cached_benign_scores
    scores_val_ref[attack_mask_ref] = predict_anomaly_scores(model, X_val_ref_s[attack_mask_ref])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores_val_ref[benign_mask_ref], bins=80, alpha=0.6, color="green", label="Benign", density=True)
    ax.hist(scores_val_ref[attack_mask_ref], bins=80, alpha=0.6, color="red", label="Attack", density=True)
    ax.axvline(opt_threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold={opt_threshold:.4f} (FPR={TARGET_FPR*100:.1f}%)")
    ax.set_title(f"OptIForest Anomaly Score Distribution — val_binary ({threshold_label} held-out)")
    ax.set_xlabel("Anomaly Score (higher = abnormal)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "optiforest_score_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {IMAGES_DIR / 'optiforest_score_dist.png'}")

    results: dict = {
        "optiforest_params": OPTIFOREST_PARAMS,
        "target_fpr": TARGET_FPR,
        "optiforest_threshold": float(opt_threshold),
        "zero_shot_per_label": {},
        "known_attack_per_label": {},
    }

    print("\nDang eval ZERO-SHOT tren 14 test_zeroshot.csv (chi attack bi giu lai, khong Benign) ...")
    for label in attack_labels:
        test_path = SPLITS_DIR / "by_label" / label / "test_zeroshot.csv"
        test_df = load_split_csv(test_path)
        X_test_raw = test_df[feature_cols].values.astype("float32")
        X_test_s = np.clip(scaler.transform(X_test_raw), -CLIP_VAL, CLIP_VAL)
        y_test = test_df["Label"].values

        scores_test = predict_anomaly_scores(model, X_test_s)
        pred_test = (scores_test >= opt_threshold).astype(int)

        metrics = evaluate_zero_shot(y_test, pred_test)
        results["zero_shot_per_label"][label] = metrics
        print(f"  {label:30s} accuracy={metrics['accuracy']:.4f}  n={metrics['n_samples']:,}")

    print("\nDang eval KNOWN-ATTACK tren 14 val_binary.csv (Benign + 13 attack known, moi held-out fold) ...")
    for label in attack_labels:
        val_path = SPLITS_DIR / "by_label" / label / "val_binary.csv"
        val_df = load_split_csv(val_path)
        X_val_raw = val_df[feature_cols].values.astype("float32")
        X_val_s = np.clip(scaler.transform(X_val_raw), -CLIP_VAL, CLIP_VAL)
        y_val = val_df["Label"].values

        b_mask = (y_val == 0)
        a_mask = ~b_mask

        scores_val = np.empty(len(y_val), dtype=np.float32)
        if len(cached_benign_scores) == b_mask.sum():
            scores_val[b_mask] = cached_benign_scores
        else:
            scores_val[b_mask] = predict_anomaly_scores(model, X_val_s[b_mask])

        scores_val[a_mask] = predict_anomaly_scores(model, X_val_s[a_mask])

        pred_val = (scores_val >= opt_threshold).astype(int)
        metrics = evaluate(y_val, pred_val)
        results["known_attack_per_label"][label] = metrics
        print(f"  held-out={label:30s} recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}  "
              f"fpr={metrics['fpr']:.4f}  n={metrics['n_samples']:,}")

    def macro_avg(per_label: dict, key: str) -> float | None:
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

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDa ghi metrics: {METRICS_PATH}")
    print(f"Artifacts: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
