from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from common.knn_config import KNN_PARAMS  # noqa: E402
from models.clad.util.metrics import best_f1_threshold, threshold_by_fpr  # noqa: E402

SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", BASE_DIR / "data" / "CICIDS2018"))
LABELS_CONFIG_PATH = Path(os.environ.get("LABELS_CONFIG_PATH", BASE_DIR / "config" / "labels.json"))

ARTIFACTS_ROOT = BASE_DIR / "artifacts" / "knn"
METRICS_PATH = ARTIFACTS_ROOT / "metrics.json"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

KNN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PyTorchKNNClassifier:
    """
    GPU-accelerated k-Nearest Neighbors classifier using PyTorch tensor operations.
    Supports distance-weighted probability output and chunked inference to fit in VRAM.
    """
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "distance",
        batch_size: int = 5000,
        device: str | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.batch_size = batch_size
        self.device = device or KNN_DEVICE
        self.X_train_cpu: np.ndarray | None = None
        self.y_train_cpu: np.ndarray | None = None
        self._X_train_gpu: torch.Tensor | None = None
        self._y_train_gpu: torch.Tensor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> PyTorchKNNClassifier:
        self.X_train_cpu = np.asarray(X, dtype=np.float32)
        self.y_train_cpu = np.asarray(y, dtype=np.float32)
        self._X_train_gpu = None
        self._y_train_gpu = None
        return self

    def _ensure_gpu(self) -> None:
        if self._X_train_gpu is None:
            self._X_train_gpu = torch.tensor(self.X_train_cpu, dtype=torch.float32, device=self.device)
            self._y_train_gpu = torch.tensor(self.y_train_cpu, dtype=torch.float32, device=self.device)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._ensure_gpu()
        X_np = np.asarray(X, dtype=np.float32)
        N = len(X_np)
        probs_1: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, N, self.batch_size):
                X_batch = torch.tensor(X_np[i : i + self.batch_size], dtype=torch.float32, device=self.device)
                dists = torch.cdist(X_batch, self._X_train_gpu)
                topk_dists, topk_indices = torch.topk(dists, k=self.n_neighbors, largest=False)
                topk_labels = self._y_train_gpu[topk_indices]

                if self.weights == "distance":
                    w = 1.0 / (topk_dists + 1e-6)
                    p1 = (w * topk_labels).sum(dim=1) / w.sum(dim=1)
                else:
                    p1 = topk_labels.mean(dim=1)
                probs_1.append(p1.cpu())

        p1_all = torch.cat(probs_1).numpy()
        p0_all = 1.0 - p1_all
        return np.column_stack([p0_all, p1_all])


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
    ckpt_dir = label_artifacts_dir / "checkpoint"
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

    max_train = int(os.environ.get("KNN_MAX_TRAIN_SAMPLES", KNN_PARAMS.get("max_train_samples", 50000)))
    if max_train > 0 and len(X_train_s) > max_train + 10000:
        X_rem, X_ref, y_rem, y_ref = train_test_split(
            X_train_s, y_train, test_size=max_train, stratify=y_train, random_state=SEED
        )
        tune_size = min(20000, len(X_rem))
        _, X_tune, _, y_tune = train_test_split(
            X_rem, y_rem, test_size=tune_size, stratify=y_rem, random_state=SEED
        )
    else:
        X_ref, X_tune, y_ref, y_tune = train_test_split(
            X_train_s, y_train, test_size=0.2, stratify=y_train, random_state=SEED
        )

    if model_path.exists():
        model = joblib.load(model_path)
    else:
        model = PyTorchKNNClassifier(
            n_neighbors=KNN_PARAMS.get("n_neighbors", 5),
            weights=KNN_PARAMS.get("weights", "distance"),
            batch_size=KNN_PARAMS.get("batch_size", 5000),
            device=KNN_DEVICE,
        )
        t0 = time.time()
        model.fit(X_ref, y_ref)
        print(f"    Indexed {len(X_ref):,} samples on {KNN_DEVICE} in {time.time()-t0:.2f}s")
        joblib.dump(model, model_path)

    TARGET_FPR = float(os.environ.get("TARGET_FPR", 0.005))
    proba_tune = model.predict_proba(X_tune)[:, 1]
    proba_tune_benign = proba_tune[y_tune == 0]
    threshold = threshold_by_fpr(proba_tune_benign, target_fpr=TARGET_FPR)
    empirical_tune_fpr = float((proba_tune_benign >= threshold).mean())
    print(f"    [Train-Tune] Iso-FPR threshold={threshold:.4f}  (Tune Benign FPR={empirical_tune_fpr*100:.3f}% "
          f"on {len(proba_tune_benign):,} samples, target={TARGET_FPR*100:.1f}%)")

    t_val = time.time()
    proba_val = model.predict_proba(X_val_s)[:, 1]
    print(f"    Val inference ({len(X_val_s):,} samples): {time.time()-t_val:.1f}s")

    proba_test = model.predict_proba(X_test_s)[:, 1]

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
    print(f"KNN model device: {KNN_DEVICE}  |  {len(feature_cols)} features  |  {len(attack_labels)} folds")

    results: dict = {"knn_params": KNN_PARAMS, "zero_shot_per_label": {}, "known_attack_per_label": {}}

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
