from __future__ import annotations

import json
import os
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

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from models.autoencoder import BenignAutoencoder  # noqa: E402
from models.clad.util.metrics import threshold_by_fpr  # noqa: E402

SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", BASE_DIR / "data" / "CICIDS2018"))
LABELS_CONFIG_PATH = Path(os.environ.get("LABELS_CONFIG_PATH", BASE_DIR / "config" / "labels.json"))

ARTIFACTS_DIR = BASE_DIR / "artifacts" / "ae_only"
IMAGES_DIR = ARTIFACTS_DIR / "images"
CKPT_DIR = ARTIFACTS_DIR / "checkpoint"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

for d in (IMAGES_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SCALER_PATH = CKPT_DIR / "scaler.pkl"
AE_CKPT_PATH = CKPT_DIR / "ae_best.pt"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)

BOTTLENECK_DIM = 16
EPOCHS = 100
BATCH_SIZE = 2048
LR = 1e-2
PATIENCE = 7
GRAD_CLIP = 1.0
CLIP_VAL = 20.0
TARGET_FPR = float(os.environ.get("TARGET_FPR", 0.005))

FEATURE_COLS: list[str] | None = None


def load_split_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "Label" in df.columns and "Label_Text" in df.columns, f"{path}: missing Label/Label_Text"
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("Label", "Label_Text")]


def get_recon_errors(model: BenignAutoencoder, X_np: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X_np, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    errors = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            errors.append(model.reconstruction_error(xb).cpu().numpy())
    return np.concatenate(errors)


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
    # test_zeroshot.csv la 100% attack (khong co Benign) nen f1/precision/fpr
    # khong co y nghia thong ke that — accuracy o day chinh la recall.
    acc = float((y_true == y_pred).mean())
    return {"accuracy": acc, "n_samples": int(len(y_true))}


def main() -> None:
    global FEATURE_COLS

    with open(LABELS_CONFIG_PATH, "r", encoding="utf-8") as f:
        labels_cfg = json.load(f)
    attack_labels = labels_cfg["attack_labels"]

    print("Dang doc train_benign_only.csv ...")
    train_df = load_split_csv(SPLITS_DIR / "train_benign_only.csv")
    FEATURE_COLS = get_feature_cols(train_df)
    print(f"  {len(train_df):,} dong Benign, {len(FEATURE_COLS)} feature.")

    X_train_raw = train_df[FEATURE_COLS].values.astype("float32")

    if SCALER_PATH.exists():
        print(f"[Checkpoint] Found {SCALER_PATH} - loading scaler.")
        scaler = joblib.load(SCALER_PATH)
    else:
        scaler = RobustScaler()
        scaler.fit(X_train_raw)
        joblib.dump(scaler, SCALER_PATH)
        print(f"RobustScaler fit tren benign train - saved to {SCALER_PATH}")

    X_train_s = np.clip(scaler.transform(X_train_raw), -CLIP_VAL, CLIP_VAL)

    idx_shuffle = np.random.permutation(len(X_train_s))
    val_size = int(len(X_train_s) * 0.1)
    X_fit_s = X_train_s[idx_shuffle[val_size:]]
    X_ae_val_s = X_train_s[idx_shuffle[:val_size]]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_fit_s, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_ae_val_s, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
    )

    ae_model = BenignAutoencoder(input_dim=len(FEATURE_COLS), bottleneck_dim=BOTTLENECK_DIM).to(DEVICE)
    optimizer = optim.Adam(ae_model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.MSELoss()

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    patience_counter = 0

    if AE_CKPT_PATH.exists():
        print(f"[Checkpoint] Found {AE_CKPT_PATH} - loading, skip training.")
        ae_model.load_state_dict(torch.load(AE_CKPT_PATH, map_location=DEVICE))
    else:
        print(f"Training Autoencoder - {EPOCHS} epochs, bottleneck={BOTTLENECK_DIM}")
        print(f"Train: {len(X_fit_s):,}  Val: {len(X_ae_val_s):,}")
        t0 = time.time()
        for epoch in range(1, EPOCHS + 1):
            ae_model.train()
            epoch_loss = 0.0
            for (xb,) in train_loader:
                xb = xb.to(DEVICE)
                x_hat, _ = ae_model(xb)
                loss = criterion(x_hat, xb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ae_model.parameters(), GRAD_CLIP)
                optimizer.step()
                epoch_loss += loss.item()
            avg_train = epoch_loss / len(train_loader)

            if np.isnan(avg_train) or avg_train > 1e10:
                print(f"  Loss no tai epoch {epoch} ({avg_train:.2e}) - dung.")
                break

            ae_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (xb,) in val_loader:
                    xb = xb.to(DEVICE)
                    x_hat, _ = ae_model(xb)
                    val_loss += criterion(x_hat, xb).item()
            avg_val = val_loss / len(val_loader)

            train_losses.append(avg_train)
            val_losses.append(avg_val)
            scheduler.step(avg_val)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{EPOCHS}  train={avg_train:.6f}  val={avg_val:.6f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(ae_model.state_dict(), AE_CKPT_PATH)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping tai epoch {epoch}")
                    break

        print(f"Training time: {time.time()-t0:.1f}s  Best val loss: {best_val_loss:.6f}")
        ae_model.load_state_dict(torch.load(AE_CKPT_PATH, map_location=DEVICE))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label="Train Loss", color="steelblue", linewidth=2)
        ax.plot(val_losses, label="Val Loss", color="tomato", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Autoencoder Training Curve (Benign Only)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "ae_training_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Da luu {IMAGES_DIR / 'ae_training_curve.png'}")

    print(f"\nDang chon threshold theo Iso-FPR (target_fpr={TARGET_FPR*100:.1f}%) tren {len(X_train_s):,} train benign ...")
    t0_score = time.time()
    errors_train_benign = get_recon_errors(ae_model, X_train_s)
    print(f"  Scored {len(X_train_s):,} train benign samples in {time.time()-t0_score:.1f}s")
    ae_threshold = threshold_by_fpr(errors_train_benign, target_fpr=TARGET_FPR)
    empirical_train_fpr = float((errors_train_benign >= ae_threshold).mean())
    print(f"  AE threshold (Quantile {1.0-TARGET_FPR:.3f}): {ae_threshold:.6f}  "
          f"(Train Benign FPR={empirical_train_fpr*100:.3f}%)")

    threshold_label = attack_labels[0]
    val_ref_df = load_split_csv(SPLITS_DIR / "by_label" / threshold_label / "val_binary.csv")
    X_val_ref_raw = val_ref_df[FEATURE_COLS].values.astype("float32")
    X_val_ref_s = np.clip(scaler.transform(X_val_ref_raw), -CLIP_VAL, CLIP_VAL)
    y_val_ref = val_ref_df["Label"].values
    errors_val_ref = get_recon_errors(ae_model, X_val_ref_s)
    cached_benign_errors = errors_val_ref[y_val_ref == 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    clip_p = np.percentile(errors_val_ref, 99)
    errs_clipped = np.clip(errors_val_ref, 0, clip_p)
    ax.hist(errs_clipped[y_val_ref == 0], bins=100, alpha=0.6, color="green", label="Benign", density=True)
    ax.hist(errs_clipped[y_val_ref == 1], bins=100, alpha=0.6, color="red", label="Attack", density=True)
    ax.axvline(min(ae_threshold, clip_p), color="black", linestyle="--", linewidth=2,
               label=f"Threshold={ae_threshold:.4f} (FPR={TARGET_FPR*100:.1f}%)")
    ax.set_title(f"AE Reconstruction Error Distribution — val_binary ({threshold_label} held-out)")
    ax.set_xlabel("MSE Reconstruction Error")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "ae_error_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Da luu {IMAGES_DIR / 'ae_error_dist.png'}")

    print("\nDang eval ZERO-SHOT tren 14 test_zeroshot.csv (chi attack bi giu lai, khong Benign) ...")
    results: dict = {
        "target_fpr": TARGET_FPR,
        "ae_threshold": ae_threshold,
        "zero_shot_per_label": {},
        "known_attack_per_label": {},
    }

    for label in attack_labels:
        test_path = SPLITS_DIR / "by_label" / label / "test_zeroshot.csv"
        test_df = load_split_csv(test_path)
        X_test_raw = test_df[FEATURE_COLS].values.astype("float32")
        X_test_s = np.clip(scaler.transform(X_test_raw), -CLIP_VAL, CLIP_VAL)
        y_test = test_df["Label"].values

        errors_test = get_recon_errors(ae_model, X_test_s)
        pred_test = (errors_test >= ae_threshold).astype(int)

        metrics = evaluate_zero_shot(y_test, pred_test)
        results["zero_shot_per_label"][label] = metrics
        print(f"  {label:30s} accuracy={metrics['accuracy']:.4f}  n={metrics['n_samples']:,}")

    print("\nDang eval KNOWN-ATTACK tren 14 val_binary.csv (Benign + 13 attack known, moi held-out fold) ...")
    for label in attack_labels:
        val_path = SPLITS_DIR / "by_label" / label / "val_binary.csv"
        val_df = load_split_csv(val_path)
        X_val_raw = val_df[FEATURE_COLS].values.astype("float32")
        X_val_s = np.clip(scaler.transform(X_val_raw), -CLIP_VAL, CLIP_VAL)
        y_val = val_df["Label"].values

        b_mask = (y_val == 0)
        a_mask = ~b_mask
        errors_val = np.empty(len(y_val), dtype=np.float32)
        if len(cached_benign_errors) == b_mask.sum():
            errors_val[b_mask] = cached_benign_errors
        else:
            errors_val[b_mask] = get_recon_errors(ae_model, X_val_s[b_mask])
        errors_val[a_mask] = get_recon_errors(ae_model, X_val_s[a_mask])
        pred_val = (errors_val >= ae_threshold).astype(int)

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
