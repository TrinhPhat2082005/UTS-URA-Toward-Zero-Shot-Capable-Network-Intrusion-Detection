from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

import os

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from models.clad.model import build_model  # noqa: E402
from models.clad.losses.clad_loss import CLADLoss  # noqa: E402
from models.clad.util.schedules import WarmupCosineSchedule  # noqa: E402
from models.clad.util.distance import get_centroid, cosine_scores_to_centroid  # noqa: E402
from models.clad.util.metrics import best_f1_threshold  # noqa: E402

SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", BASE_DIR / "data" / "CICIDS2018"))
LABELS_CONFIG_PATH = Path(os.environ.get("LABELS_CONFIG_PATH", BASE_DIR / "config" / "labels.json"))

ARTIFACTS_ROOT = BASE_DIR / "artifacts" / "clad"
METRICS_PATH = ARTIFACTS_ROOT / "metrics.json"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

CLIP_VAL = 20.0

# Matches github.com/jackwilkie/CLOSR/train_clad.py's actual values, not the
# paper text (e.g. warmup_steps = (epochs // 10) * steps_per_epoch below,
# not min(20, epochs)).
HIDDEN_DIM = 512
N_LAYERS = 4
D_OUT = 32
DROPOUT = 0.0
PEAK_LR = 4.4e-6
WEIGHT_DECAY = 3e-7
MARGIN = 1.0
ALPHA = 0.5
BATCH_SIZE = 4096
EPOCHS = 200

EVAL_EVERY = 10
PATIENCE_EVALS = 2
RESUME_SAVE_EVERY = 10


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
    # test_zeroshot.csv la 100% attack (khong co Benign) nen f1/precision/fpr
    # khong co y nghia thong ke that — accuracy o day chinh la recall.
    acc = float((y_true == y_pred).mean())
    return {"accuracy": acc, "n_samples": int(len(y_true))}


@torch.no_grad()
def get_embeddings(model, X: np.ndarray, device, batch_size: int = 8192) -> torch.Tensor:
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    zs = []
    for (xb,) in loader:
        xb = xb.to(device)
        zs.append(model(xb).cpu())
    return torch.cat(zs)


def train_one_label(label: str, feature_cols: list[str], device: torch.device,
                     epochs: int, resume: bool) -> tuple[dict, dict]:
    label_artifacts_dir = ARTIFACTS_ROOT / label
    ckpt_dir = label_artifacts_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = ckpt_dir / "scaler.pkl"
    model_path = ckpt_dir / "clad_best.pt"
    meta_path = ckpt_dir / "meta.json"
    resume_path = ckpt_dir / "resume_state.pt"

    train_df = load_split_csv(SPLITS_DIR / "by_label" / label / "train_benign_attack.csv")
    val_df = load_split_csv(SPLITS_DIR / "by_label" / label / "val_binary.csv")
    test_df = load_split_csv(SPLITS_DIR / "by_label" / label / "test_zeroshot.csv")

    X_train_raw = train_df[feature_cols].values.astype("float32")
    y_train = train_df["Label"].values.astype("int64")
    X_val_raw = val_df[feature_cols].values.astype("float32")
    y_val = val_df["Label"].values.astype("int64")
    X_test_raw = test_df[feature_cols].values.astype("float32")
    y_test = test_df["Label"].values.astype("int64")

    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        scaler = RobustScaler()
        scaler.fit(X_train_raw)
        joblib.dump(scaler, scaler_path)

    X_train_s = np.clip(scaler.transform(X_train_raw), -CLIP_VAL, CLIP_VAL).astype("float32")
    X_val_s = np.clip(scaler.transform(X_val_raw), -CLIP_VAL, CLIP_VAL).astype("float32")
    X_test_s = np.clip(scaler.transform(X_test_raw), -CLIP_VAL, CLIP_VAL).astype("float32")

    d_in = X_train_s.shape[1]
    model = build_model(d_in=d_in, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
                        d_out=D_OUT, dropout=DROPOUT, block_norm=False).to(device)

    class_sample_count = np.bincount(y_train)
    class_sample_count[class_sample_count == 0] = 1
    class_weight = 1.0 / class_sample_count
    sample_weight = class_weight[y_train]
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(sample_weight).double(), num_samples=len(sample_weight), replacement=True,
    )
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, sampler=sampler, drop_last=True,
    )

    if model_path.exists() and resume:
        print(f"  [Checkpoint] Found {model_path} - loading, skip training.")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        centroid = torch.tensor(meta["centroid"], dtype=torch.float32)
        threshold = meta["threshold"]
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6,
                                      betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
        steps_per_epoch = len(train_loader)
        total_steps = epochs * steps_per_epoch
        warmup_steps = max(1, (epochs // 10) * steps_per_epoch)
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps,
                                         total_steps=total_steps, init_lr=1e-6,
                                         peak_lr=PEAK_LR, min_lr=1e-6)
        criterion = CLADLoss(m=MARGIN, alpha=ALPHA)

        start_epoch = 1
        best_f1_seen = -1.0
        evals_since_improve = 0
        stopped_early = False

        if resume_path.exists():
            state = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler._step = state["scheduler_step"]
            start_epoch = state["epoch"] + 1
            best_f1_seen = state.get("best_f1_seen", -1.0)
            evals_since_improve = state.get("evals_since_improve", 0)
            print(f"  [Resume] Tiep tuc fold {label} tu epoch {start_epoch} "
                  f"(best_f1_seen={best_f1_seen:.4f})")

        t0 = time.time()
        epoch = start_epoch - 1
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            total_loss, total_frac, n_batches = 0.0, 0.0, 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                z = model(X_batch)
                loss, frac_pos = criterion(z, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                total_frac += frac_pos
                n_batches += 1
            if epoch % 20 == 0 or epoch == epochs or epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"    Epoch {epoch:3d}/{epochs}  loss={total_loss/n_batches:.6f}  "
                      f"frac_active={total_frac/n_batches:.3f}  lr={lr_now:.2e}")

            if epoch % EVAL_EVERY == 0 or epoch == epochs:
                benign_mask_chk = (y_train == 0)
                z_train_benign_chk = get_embeddings(model, X_train_s[benign_mask_chk], device)
                centroid_chk = get_centroid(z_train_benign_chk)
                z_val_chk = get_embeddings(model, X_val_s, device)
                anom_val_chk = -cosine_scores_to_centroid(z_val_chk, centroid_chk)
                _, f1_chk, _, _ = best_f1_threshold(y_val, anom_val_chk)
                print(f"    [EarlyStop check] epoch={epoch}  val_F1={f1_chk:.4f}  "
                      f"best_so_far={max(best_f1_seen, f1_chk):.4f}")
                if f1_chk > best_f1_seen:
                    best_f1_seen = f1_chk
                    evals_since_improve = 0
                else:
                    evals_since_improve += 1
                if evals_since_improve >= PATIENCE_EVALS:
                    print(f"    [EarlyStop] Dung som tai epoch {epoch} "
                          f"(val_F1 khong cai thien sau {PATIENCE_EVALS} lan kiem tra)")
                    stopped_early = True

            if epoch % RESUME_SAVE_EVERY == 0 or epoch == epochs or stopped_early:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler_step": scheduler.step,
                    "epoch": epoch,
                    "best_f1_seen": best_f1_seen,
                    "evals_since_improve": evals_since_improve,
                }, resume_path)

            if stopped_early:
                break
        print(f"    Training time: {time.time()-t0:.1f}s ({epoch}/{epochs} epoch, "
              f"{'early-stopped' if stopped_early else 'full'})")

        benign_mask = (y_train == 0)
        z_train_benign = get_embeddings(model, X_train_s[benign_mask], device)
        centroid = get_centroid(z_train_benign)

        z_val = get_embeddings(model, X_val_s, device)
        sim_val = cosine_scores_to_centroid(z_val, centroid)
        anom_val = -sim_val
        threshold, f1_val, prec_val, rec_val = best_f1_threshold(y_val, anom_val)
        threshold = float(threshold)
        print(f"    [Val] best-F1 threshold={threshold:.4f}  F1={f1_val:.4f}  "
              f"P={prec_val:.4f}  R={rec_val:.4f}")

        if resume_path.exists():
            resume_path.unlink()

        torch.save(model.state_dict(), model_path)
        meta_path.write_text(json.dumps({
            "centroid": centroid.numpy().tolist(),
            "threshold": threshold,
        }), encoding="utf-8")

    z_test = get_embeddings(model, X_test_s, device)
    sim_test = cosine_scores_to_centroid(z_test, centroid)
    anom_test = -sim_test
    pred_test = (anom_test >= threshold).astype(int)
    zero_shot_metrics = evaluate_zero_shot(y_test, pred_test)

    z_val = get_embeddings(model, X_val_s, device)
    sim_val = cosine_scores_to_centroid(z_val, centroid)
    anom_val = -sim_val
    pred_val = (anom_val >= threshold).astype(int)
    known_metrics = evaluate(y_val, pred_val)

    return zero_shot_metrics, known_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  epochs/fold={args.epochs}  |  warmup={args.epochs // 10} epoch")

    with open(LABELS_CONFIG_PATH, "r", encoding="utf-8") as f:
        labels_cfg = json.load(f)
    attack_labels = labels_cfg["attack_labels"]

    sample_df = load_split_csv(SPLITS_DIR / "by_label" / attack_labels[0] / "train_benign_attack.csv")
    feature_cols = get_feature_cols(sample_df)
    print(f"{len(feature_cols)} feature goc, {len(attack_labels)} fold leave-one-attack-out")

    results: dict = {
        "hyperparams": {
            "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "d_out": D_OUT,
            "peak_lr": PEAK_LR, "margin": MARGIN, "alpha": ALPHA,
            "epochs": args.epochs, "batch_size": BATCH_SIZE,
            "warmup_epochs": args.epochs // 10,
        },
        "zero_shot_per_label": {}, "known_attack_per_label": {},
    }

    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    PARTIAL_METRICS_PATH = ARTIFACTS_ROOT / "metrics_partial.json"

    for label in attack_labels:
        print(f"\n=== held-out = {label} ===")
        zero_shot_metrics, known_metrics = train_one_label(label, feature_cols, device, args.epochs, args.resume)
        results["zero_shot_per_label"][label] = zero_shot_metrics
        results["known_attack_per_label"][label] = known_metrics
        print(f"  zero-shot    accuracy={zero_shot_metrics['accuracy']:.4f}")
        print(f"  known-attack recall={known_metrics['recall']:.4f}  f1={known_metrics['f1']:.4f}  "
              f"precision={known_metrics['precision']:.4f}  fpr={known_metrics['fpr']:.4f}")

        with open(PARTIAL_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "hyperparams": results["hyperparams"],
                "completed_labels": list(results["zero_shot_per_label"].keys()),
                "zero_shot_per_label": results["zero_shot_per_label"],
                "known_attack_per_label": results["known_attack_per_label"],
            }, f, indent=2, ensure_ascii=False)
        print(f"  [Partial] Da ghi {PARTIAL_METRICS_PATH} ({len(results['zero_shot_per_label'])}/{len(attack_labels)} fold)")

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

    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDa ghi metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
