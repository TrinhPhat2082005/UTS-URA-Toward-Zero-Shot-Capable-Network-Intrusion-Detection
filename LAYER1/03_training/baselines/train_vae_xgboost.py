from __future__ import annotations

import argparse
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
import torch
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from common.vae_config import VAE_PARAMS
from common.xgb_config_mono import (
    XGB_PARAMS,
    XGB_SCALE_POS_DIVISOR,
    XGB_MONOTONE_AUGMENTED,
    build_monotone_constraints,
)
from models.clad.util.metrics import best_f1_threshold, threshold_by_fpr
from models.vae import BenignVAE

SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", BASE_DIR / "data" / "CICIDS2018"))
LABELS_CONFIG_PATH = Path(os.environ.get("LABELS_CONFIG_PATH", BASE_DIR / "config" / "labels.json"))

VAE_ONLY_CKPT_DIR = BASE_DIR / "artifacts" / "vae_only" / "checkpoint"
VAE_SCALER_PATH = VAE_ONLY_CKPT_DIR / "scaler.pkl"
VAE_MODEL_PATH = VAE_ONLY_CKPT_DIR / "vae_best.pt"

ARTIFACTS_ROOT = BASE_DIR / "artifacts" / "vae_xgboost"
METRICS_PATH = ARTIFACTS_ROOT / "metrics.json"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

LATENT_DIM = VAE_PARAMS.get("latent_dim", 32)
HIDDEN_DIMS = VAE_PARAMS.get("hidden_dims", (512, 256))
BETA = VAE_PARAMS.get("beta", 0.1)
CLIP_VAL = VAE_PARAMS.get("clip_val", 20.0)

# Tuned XGBoost hyperparameters for VAE Early Fusion:
# 1. colsample_bytree = 1.0 (ensure every tree sees the anomaly score)
# 2. max_depth = 5 (shallower trees to prevent overfitting 32 latent features)
# 3. min_child_weight = 10 (prevent memorization of small training attack clusters)
# 4. reg_alpha = 1.0 (L1 regularization to enforce sparsity on noisy latent dims)
VAE_XGB_PARAMS = dict(XGB_PARAMS)
VAE_XGB_PARAMS.update(
    colsample_bytree=1.0,
    max_depth=5,
    min_child_weight=10,
    reg_alpha=1.0,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XGB_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    # test_zeroshot.csv is 100% attack (no Benign)
    acc = float((y_true == y_pred).mean())
    return {"accuracy": acc, "n_samples": int(len(y_true))}


def get_vae_features(
    vae_model: BenignVAE,
    X_scaled: np.ndarray,
    batch_size: int = 4096,
    beta: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    vae_model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    latents, scores = [], []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            mu, logvar = vae_model.encode(xb)
            x_hat = vae_model.decode(mu)
            recon_err = ((xb - x_hat) ** 2).mean(dim=1)
            kl_err = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) / vae_model.input_dim
            sc = recon_err + beta * kl_err

            latents.append(mu.cpu().numpy())
            scores.append(sc.cpu().numpy())

    return np.concatenate(latents), np.concatenate(scores)


def train_one_label(label: str, feature_cols: list[str], vae_model: BenignVAE, vae_scaler) -> dict:
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

    X_train_xgb_s = np.clip(vae_scaler.transform(X_train_raw), -CLIP_VAL, CLIP_VAL)
    X_val_xgb_s = np.clip(vae_scaler.transform(X_val_raw), -CLIP_VAL, CLIP_VAL)
    X_test_xgb_s = np.clip(vae_scaler.transform(X_test_raw), -CLIP_VAL, CLIP_VAL)

    z_train, sc_train = get_vae_features(vae_model, X_train_xgb_s, beta=BETA)
    z_val, sc_val = get_vae_features(vae_model, X_val_xgb_s, beta=BETA)
    z_test, sc_test = get_vae_features(vae_model, X_test_xgb_s, beta=BETA)

    X_train_aug = np.concatenate([X_train_xgb_s, z_train, sc_train.reshape(-1, 1)], axis=1)
    X_val_aug = np.concatenate([X_val_xgb_s, z_val, sc_val.reshape(-1, 1)], axis=1)
    X_test_aug = np.concatenate([X_test_xgb_s, z_test, sc_test.reshape(-1, 1)], axis=1)

    scale_pos = float((y_train == 0).sum()) / float((y_train == 1).sum())

    # Monotone constraint: column order [78 raw | 32 vae_latent | 1 vae_anomaly_score] = 111 cols.
    # Only vae_anomaly_score is constrained with +1 (higher score => more abnormal => higher P(attack));
    # 32 vae_latent and 78 base features remain unconstrained (0).
    n_base = len(feature_cols)
    mono = build_monotone_constraints(
        n_base=n_base,
        n_augmented_free=LATENT_DIM,
        n_augmented_monotone=1,
    )
    assert mono is None or len(mono) == X_train_aug.shape[1], (
        f"monotone_constraints length {len(mono)} != column count {X_train_aug.shape[1]}"
    )

    model = None
    if model_path.exists():
        try:
            loaded_model = joblib.load(model_path)
            if hasattr(loaded_model, "n_features_in_") and loaded_model.n_features_in_ == X_train_aug.shape[1]:
                print(f"    [Checkpoint] Loading existing model from {model_path}")
                model = loaded_model
            else:
                print(f"    [Checkpoint] Feature count mismatch ({getattr(loaded_model, 'n_features_in_', None)} != {X_train_aug.shape[1]}) — retraining.")
        except Exception:
            model = None

    if model is None:
        model = xgb.XGBClassifier(
            **VAE_XGB_PARAMS,
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
            augmented_feature_names = (
                feature_cols
                + [f"vae_latent_{i}" for i in range(LATENT_DIM)]
                + ["vae_anomaly_score"]
            )
            shap_idx = np.random.choice(len(X_val_aug), min(2000, len(X_val_aug)), replace=False)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val_aug[shap_idx])
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top15_idx = np.argsort(mean_abs_shap)[::-1][:15]
            top15_features = [augmented_feature_names[i] for i in top15_idx]
            colors = ["#e74c3c" if "vae_" in f else "#3498db" for f in top15_features]

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(range(len(top15_features)), mean_abs_shap[top15_idx][::-1], color=colors[::-1])
            ax.set_yticks(range(len(top15_features)))
            ax.set_yticklabels(top15_features[::-1], fontsize=9)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"VAE+XGBoost SHAP Top 15 — held-out={label}\nRed = VAE feature, Blue = original")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(shap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except (ImportError, Exception) as e:
            print(f"    [SHAP] Skipped plot: {e}")

    return {
        "threshold_from_train": threshold,
        "proba_val": proba_val,
        "y_val": y_val,
        "proba_test": proba_test,
        "y_test": y_test,
    }


def run_unit_test() -> None:
    print("=== Running Unit Test for VAE-XGBoost (32 latent dims, beta=0.1) ===")

    print(f"1. Checking VAE checkpoint directory at {VAE_ONLY_CKPT_DIR} ...")
    assert VAE_SCALER_PATH.exists(), f"Scaler not found at {VAE_SCALER_PATH}"
    vae_scaler = joblib.load(VAE_SCALER_PATH)
    print("   [OK] Scaler loaded.")

    print(f"2. Checking VAE model architecture (latent_dim={LATENT_DIM}) ...")
    sample_df = load_split_csv(SPLITS_DIR / "train_benign_only.csv")
    feature_cols = get_feature_cols(sample_df)
    n_base = len(feature_cols)
    assert n_base == 78, f"Expected 78 base features, got {n_base}"

    vae_model = BenignVAE(
        input_dim=n_base,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        dropout=VAE_PARAMS.get("dropout", 0.1),
        negative_slope=VAE_PARAMS.get("negative_slope", 0.2),
    ).to(DEVICE)

    if VAE_MODEL_PATH.exists():
        try:
            vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
            print(f"   [OK] Loaded trained weights from {VAE_MODEL_PATH}")
        except Exception as e:
            print(f"   [Notice] Existing {VAE_MODEL_PATH} incompatible ({e}) — using initialized 32-dim weights for test.")
    else:
        print(f"   [Notice] {VAE_MODEL_PATH} not found yet — using initialized 32-dim weights for test.")
    vae_model.eval()

    print("3. Testing VAE feature extraction on synthetic batch (batch_size=64) ...")
    X_dummy = np.random.randn(64, n_base).astype("float32")
    X_dummy_s = np.clip(vae_scaler.transform(X_dummy), -CLIP_VAL, CLIP_VAL)

    z, sc = get_vae_features(vae_model, X_dummy_s, batch_size=32, beta=BETA)
    assert z.shape == (64, LATENT_DIM), f"Expected latent shape (64, {LATENT_DIM}), got {z.shape}"
    assert sc.shape == (64,), f"Expected score shape (64,), got {sc.shape}"
    print(f"   [OK] Latents shape: {z.shape}, Anomaly scores shape: {sc.shape}")

    print("4. Testing feature concatenation ...")
    X_aug = np.concatenate([X_dummy_s, z, sc.reshape(-1, 1)], axis=1)
    expected_cols = n_base + LATENT_DIM + 1
    assert X_aug.shape == (64, expected_cols), (
        f"Expected shape (64, {expected_cols}), got {X_aug.shape}"
    )
    print(f"   [OK] Augmented feature matrix shape: {X_aug.shape} (78 base + {LATENT_DIM} latent + 1 anomaly_score = {expected_cols} total)")

    print("5. Testing Monotone Constraints vector ...")
    mono = build_monotone_constraints(
        n_base=n_base,
        n_augmented_free=LATENT_DIM,
        n_augmented_monotone=1,
    )
    assert mono is not None, "Monotone constraints should not be None"
    assert len(mono) == expected_cols, f"Constraint length {len(mono)} != {expected_cols}"
    assert mono[-1] == 1, "Last column must be constrained (+1)"
    assert sum(mono) == 1, "Only 1 column should be constrained"
    assert all(m == 0 for m in mono[:-1]), "Base and latent columns must be 0"
    print("   [OK] Monotone constraints verified (length 111): (0, 0, ...) ... (0, 0, 1)")

    print("6. Testing XGBoost training with updated hyperparameters on mini dataset ...")
    print(f"   XGBoost config: max_depth={VAE_XGB_PARAMS['max_depth']}, "
          f"colsample_bytree={VAE_XGB_PARAMS['colsample_bytree']}, "
          f"min_child_weight={VAE_XGB_PARAMS['min_child_weight']}, "
          f"reg_alpha={VAE_XGB_PARAMS['reg_alpha']}")
    y_dummy = np.array([0] * 32 + [1] * 32)
    test_params = dict(VAE_XGB_PARAMS)
    test_params["n_estimators"] = 5
    test_params["early_stopping_rounds"] = None

    test_xgb = xgb.XGBClassifier(
        **test_params,
        scale_pos_weight=1.0,
        monotone_constraints=mono,
        device=XGB_DEVICE,
    )
    test_xgb.fit(X_aug, y_dummy)
    preds = test_xgb.predict_proba(X_aug)[:, 1]
    assert preds.shape == (64,), f"Expected predictions shape (64,), got {preds.shape}"
    print("   [OK] XGBoost fit and inference successful.")

    print("\n[SUCCESS] All VAE-XGBoost unit tests passed cleanly with 32-dim latent space and updated XGB hyperparameters!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAE-XGBoost Model (Layer 1 LOAO)")
    parser.add_argument("--unit-test", action="store_true", help="Run quick unit test and exit")
    args = parser.parse_args()

    if args.unit_test:
        run_unit_test()
        return

    with open(LABELS_CONFIG_PATH, "r", encoding="utf-8") as f:
        labels_cfg = json.load(f)
    attack_labels = labels_cfg["attack_labels"]

    assert VAE_SCALER_PATH.exists() and VAE_MODEL_PATH.exists(), (
        f"VAE checkpoint not found at {VAE_ONLY_CKPT_DIR} — train the 32-dim benign-only VAE first via train_vae_only.py."
    )
    print(f"Loading VAE checkpoint from {VAE_ONLY_CKPT_DIR} ...")
    vae_scaler = joblib.load(VAE_SCALER_PATH)

    sample_df = load_split_csv(SPLITS_DIR / "by_label" / attack_labels[0] / "train_benign_attack.csv")
    feature_cols = get_feature_cols(sample_df)

    vae_model = BenignVAE(
        input_dim=len(feature_cols),
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        dropout=VAE_PARAMS.get("dropout", 0.1),
        negative_slope=VAE_PARAMS.get("negative_slope", 0.2),
    ).to(DEVICE)
    vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
    vae_model.eval()

    print(f"XGBoost device: {XGB_DEVICE}  |  {len(feature_cols)} base features + {LATENT_DIM} latent + 1 anomaly_score = {len(feature_cols) + LATENT_DIM + 1} features")
    print(f"XGBoost tuned params: colsample_bytree={VAE_XGB_PARAMS['colsample_bytree']}, "
          f"max_depth={VAE_XGB_PARAMS['max_depth']}, min_child_weight={VAE_XGB_PARAMS['min_child_weight']}, "
          f"reg_alpha={VAE_XGB_PARAMS['reg_alpha']}")
    print(f"Monotone constraint: {'ON' if XGB_MONOTONE_AUGMENTED else 'OFF'} "
          f"(only vae_anomaly_score constrained (+1); {LATENT_DIM} latents unconstrained)")

    results: dict = {
        "xgb_params": VAE_XGB_PARAMS,
        "vae_params": VAE_PARAMS,
        "monotone_augmented": XGB_MONOTONE_AUGMENTED,
        "monotone_n_constrained": 1 if XGB_MONOTONE_AUGMENTED else 0,
        "monotone_note": f"only vae_anomaly_score constrained (+1); {LATENT_DIM} vae_latent free",
        "scale_pos_divisor": XGB_SCALE_POS_DIVISOR,
        "zero_shot_per_label": {},
        "known_attack_per_label": {},
    }

    raw_results: dict[str, dict] = {}
    for label in attack_labels:
        print(f"\n=== held-out = {label} ===")
        raw_results[label] = train_one_label(label, feature_cols, vae_model, vae_scaler)

    threshold_per_label = {label: float(r["threshold_from_train"]) for label, r in raw_results.items()}
    threshold_mean = float(np.mean(list(threshold_per_label.values())))
    print(f"\n=== Mean Threshold (from 14 folds, tuned on train) = {threshold_mean:.4f} ===")
    print(f"    Per-fold thresholds: { {k: round(v, 4) for k, v in threshold_per_label.items()} }")

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

    print("\n=== SUMMARY (macro avg over 14 held-out folds) ===")
    zs = results["summary"]["zero_shot_macro_avg"]
    ka = results["summary"]["known_attack_macro_avg"]
    print(f"  Zero-shot   : accuracy={zs['accuracy']:.4f}")
    print(f"  Known-attack: f1={ka['f1']:.4f}  recall={ka['recall']:.4f}  precision={ka['precision']:.4f}  "
          f"accuracy={ka['accuracy']:.4f}  fpr={ka['fpr']:.4f}")

    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
