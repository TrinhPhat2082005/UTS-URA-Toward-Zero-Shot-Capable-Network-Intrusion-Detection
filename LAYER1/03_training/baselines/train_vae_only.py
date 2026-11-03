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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from common.vae_config import VAE_PARAMS
from models.clad.util.metrics import best_f1_threshold, threshold_by_fpr
from models.vae import BenignVAE

SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", BASE_DIR / "data" / "CICIDS2018"))
LABELS_CONFIG_PATH = Path(os.environ.get("LABELS_CONFIG_PATH", BASE_DIR / "config" / "labels.json"))

ARTIFACTS_DIR = BASE_DIR / "artifacts" / "vae_only"
IMAGES_DIR = ARTIFACTS_DIR / "images"
CKPT_DIR = ARTIFACTS_DIR / "checkpoint"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

for d in (IMAGES_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SCALER_PATH = CKPT_DIR / "scaler.pkl"
VAE_CKPT_PATH = CKPT_DIR / "vae_best.pt"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

HIDDEN_DIMS = VAE_PARAMS.get("hidden_dims", (512, 256))
LATENT_DIM = VAE_PARAMS.get("latent_dim", 64)
BETA = VAE_PARAMS.get("beta", 0.01)
EPOCHS = VAE_PARAMS.get("epochs", 100)
BATCH_SIZE = VAE_PARAMS.get("batch_size", 2048)
LR = VAE_PARAMS.get("learning_rate", 0.001)
PATIENCE = VAE_PARAMS.get("patience", 7)
GRAD_CLIP = VAE_PARAMS.get("grad_clip", 1.0)
CLIP_VAL = VAE_PARAMS.get("clip_val", 20.0)


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


def get_anomaly_scores(
    model: BenignVAE, X: np.ndarray, batch_size: int = 4096, beta: float = 0.01
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=DEVICE)
            s = model.anomaly_score(xb, beta=beta)
            scores.append(s.cpu().numpy())
    return np.concatenate(scores)


def main() -> None:
    with open(LABELS_CONFIG_PATH, "r", encoding="utf-8") as f:
        labels_cfg = json.load(f)
    attack_labels = labels_cfg["attack_labels"]

    print(f"=== VAE Baseline (Zavrak & Iskefiyeli, 2020 architecture) ===")
    print(f"Device: {DEVICE}  |  Params: {VAE_PARAMS}")

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

    # 90% train / 10% val on Benign for early stopping
    n_val = int(len(X_train_s) * 0.10)
    indices = np.random.permutation(len(X_train_s))
    val_idx, fit_idx = indices[:n_val], indices[n_val:]
    X_fit_s = X_train_s[fit_idx]
    X_val_s = X_train_s[val_idx]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_fit_s)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_s)), batch_size=BATCH_SIZE, shuffle=False)

    vae_model = BenignVAE(
        input_dim=len(feature_cols),
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        dropout=VAE_PARAMS.get("dropout", 0.1),
        negative_slope=VAE_PARAMS.get("negative_slope", 0.2),
    ).to(DEVICE)

    optimizer = optim.Adam(vae_model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    patience_counter = 0

    should_train = True
    if VAE_CKPT_PATH.exists():
        try:
            ckpt = torch.load(VAE_CKPT_PATH, map_location=DEVICE)
            vae_model.load_state_dict(ckpt)
            print(f"[Checkpoint] Found compatible {VAE_CKPT_PATH} (latent_dim={LATENT_DIM}) - loading, skip training.")
            should_train = False
        except Exception as e:
            print(f"[Checkpoint] Incompatible checkpoint for latent_dim={LATENT_DIM} ({e}) - will train anew.")
            should_train = True

    if not should_train:
        pass
    else:
        print(f"Training VAE - {EPOCHS} epochs, hidden={HIDDEN_DIMS}, latent={LATENT_DIM}, beta={BETA}")
        print(f"Train: {len(X_fit_s):,}  Val: {len(X_val_s):,}")
        t0 = time.time()

        for epoch in range(1, EPOCHS + 1):
            vae_model.train()
            epoch_loss = 0.0
            recon_epoch = 0.0
            kl_epoch = 0.0

            for (xb,) in train_loader:
                xb = xb.to(DEVICE)
                x_hat, mu, logvar = vae_model(xb)
                loss, r_loss, k_loss = vae_model.loss_function(xb, x_hat, mu, logvar, beta=BETA)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(vae_model.parameters(), GRAD_CLIP)
                optimizer.step()

                epoch_loss += loss.item() * len(xb)
                recon_epoch += r_loss.item() * len(xb)
                kl_epoch += k_loss.item() * len(xb)

            epoch_loss /= len(X_fit_s)
            recon_epoch /= len(X_fit_s)
            kl_epoch /= len(X_fit_s)
            train_losses.append(epoch_loss)

            vae_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (xb,) in val_loader:
                    xb = xb.to(DEVICE)
                    x_hat, mu, logvar = vae_model(xb)
                    loss, _, _ = vae_model.loss_function(xb, x_hat, mu, logvar, beta=BETA)
                    val_loss += loss.item() * len(xb)
            val_loss /= len(X_val_s)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{EPOCHS} | Train: {epoch_loss:.5f} (Recon={recon_epoch:.5f}, KL={kl_epoch:.3f}) | Val: {val_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(vae_model.state_dict(), VAE_CKPT_PATH)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping tai epoch {epoch} (patience={PATIENCE})")
                    break

        print(f"Training time: {time.time()-t0:.1f}s  Best val loss: {best_val_loss:.6f}")
        vae_model.load_state_dict(torch.load(VAE_CKPT_PATH, map_location=DEVICE))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label="Train Loss (ELBO)", color="steelblue", linewidth=2)
        ax.plot(val_losses, label="Val Loss (ELBO)", color="tomato", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss (MSE + beta*KL)")
        ax.set_title("VAE Training Curve (Benign Only)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "vae_training_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {IMAGES_DIR / 'vae_training_curve.png'}")

    TARGET_FPR = float(os.environ.get("TARGET_FPR", 0.005))
    print(f"\nDang chon threshold theo Iso-FPR (target_fpr={TARGET_FPR*100:.1f}%) tren {len(X_train_s):,} train benign ...")
    t0_score = time.time()
    scores_train_benign = get_anomaly_scores(vae_model, X_train_s, beta=BETA)
    print(f"  Scored {len(X_train_s):,} train benign samples in {time.time()-t0_score:.1f}s")
    vae_threshold = threshold_by_fpr(scores_train_benign, target_fpr=TARGET_FPR)
    empirical_train_fpr = float((scores_train_benign >= vae_threshold).mean())
    print(f"  VAE threshold (Quantile {1.0-TARGET_FPR:.3f}): {vae_threshold:.6f}  "
          f"(Train Benign FPR={empirical_train_fpr*100:.3f}%)")

    # Benign validation samples are identical across all 14 val_binary.csv splits
    threshold_label = attack_labels[0]
    val_ref_df = load_split_csv(SPLITS_DIR / "by_label" / threshold_label / "val_binary.csv")
    X_val_ref_raw = val_ref_df[feature_cols].values.astype("float32")
    X_val_ref_s = np.clip(scaler.transform(X_val_ref_raw), -CLIP_VAL, CLIP_VAL)
    y_val_ref = val_ref_df["Label"].values
    scores_val_ref = get_anomaly_scores(vae_model, X_val_ref_s, beta=BETA)
    cached_benign_scores = scores_val_ref[y_val_ref == 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    clip_p = np.percentile(scores_val_ref, 99)
    sc = np.clip(scores_val_ref, 0, clip_p)
    ax.hist(sc[y_val_ref == 0], bins=100, alpha=0.6, color="green", label="Benign", density=True)
    ax.hist(sc[y_val_ref == 1], bins=100, alpha=0.6, color="red", label="Attack", density=True)
    ax.axvline(min(vae_threshold, clip_p), color="black", linestyle="--", linewidth=2,
               label=f"Threshold={vae_threshold:.4f} (FPR={TARGET_FPR*100:.1f}%)")
    ax.set_title(f"VAE Anomaly Score Distribution — val_binary ({threshold_label} held-out)")
    ax.set_xlabel("Anomaly Score (MSE + beta*KL)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "vae_error_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {IMAGES_DIR / 'vae_error_dist.png'}")

    results: dict = {
        "vae_params": VAE_PARAMS,
        "target_fpr": TARGET_FPR,
        "vae_threshold": float(vae_threshold),
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

        scores_test = get_anomaly_scores(vae_model, X_test_s, beta=BETA)
        pred_test = (scores_test >= vae_threshold).astype(int)

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
            scores_val[b_mask] = get_anomaly_scores(vae_model, X_val_s[b_mask], beta=BETA)
        scores_val[a_mask] = get_anomaly_scores(vae_model, X_val_s[a_mask], beta=BETA)

        pred_val = (scores_val >= vae_threshold).astype(int)

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
