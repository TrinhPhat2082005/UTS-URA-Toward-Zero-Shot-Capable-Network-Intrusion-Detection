"""
Train CLAD on CIC-IDS-2018 benign traffic.

Usage (from repo root or from closr/ folder):
    python closr/train.py                          # defaults
    python closr/train.py --data_dir data          # use data/ split
    python closr/train.py --data_dir 2M_Benign_500K_attack --epochs 200
    python closr/train.py --epochs 5               # quick smoke test

The script trains on benign-only (train_isolation_benign.csv),
evaluates AUC-ROC on val_binary.csv every --eval_every epochs,
and saves the best checkpoint to closr/checkpoints/best.pt.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# Allow running from repo root OR from inside closr/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from closr.data.dataset    import load_splits, make_train_loader, make_eval_loader
from closr.model           import build_model
from closr.losses.clad_loss import CLADLoss
from closr.util.schedules  import WarmupCosineSchedule
from closr.util.checkpoint import save_checkpoint
from closr.util.distance   import get_features, get_centroid, cosine_scores_to_centroid
from closr.util.metrics    import best_f1_threshold, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',    default='binary_split_zeroshot_dataset',
                   help='Path to folder with the 4 CSV splits')
    p.add_argument('--epochs',      type=int,   default=200)
    p.add_argument('--batch_size',  type=int,   default=4096)
    p.add_argument('--train_file',  default='train_xgboost_binary.csv',
                   help='CLAD trains on benign+attack (NOT benign-only)')
    p.add_argument('--hidden_dim',  type=int,   default=512)
    p.add_argument('--n_layers',    type=int,   default=4)
    p.add_argument('--d_out',       type=int,   default=32)
    p.add_argument('--dropout',     type=float, default=0.0)
    p.add_argument('--peak_lr',     type=float, default=1e-3)
    p.add_argument('--weight_decay',type=float, default=1e-4)
    p.add_argument('--margin',      type=float, default=1.0)
    p.add_argument('--alpha',       type=float, default=0.5)
    p.add_argument('--warmup_frac', type=float, default=0.10,
                   help='Fraction of total steps used for LR warmup')
    p.add_argument('--eval_every',  type=int,   default=10,
                   help='Evaluate on val set every N epochs')
    p.add_argument('--ckpt_dir',    default='closr/checkpoints')
    p.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test, _ = load_splits(
        args.data_dir, train_file=args.train_file)
    train_loader = make_train_loader(X_train, y_train, args.batch_size)
    val_loader   = make_eval_loader(X_val,   y_val,   args.batch_size)

    d_in = X_train.shape[1]
    steps_per_epoch = len(train_loader)
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = int(args.warmup_frac * total_steps)

    print(f"Features     : {d_in}")
    print(f"Steps/epoch  : {steps_per_epoch}  |  Total steps: {total_steps}")
    print(f"Warmup steps : {warmup_steps}")

    # ── Model + optimiser ─────────────────────────────────────────────
    model = build_model(d_in=d_in, hidden_dim=args.hidden_dim,
                        n_layers=args.n_layers, d_out=args.d_out,
                        dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-6,
                                  betas=(0.9, 0.999),
                                  weight_decay=args.weight_decay)
    scheduler = WarmupCosineSchedule(optimizer,
                                     warmup_steps=warmup_steps,
                                     total_steps=total_steps,
                                     init_lr=1e-6,
                                     peak_lr=args.peak_lr,
                                     min_lr=1e-6)
    criterion = CLADLoss(m=args.margin, alpha=args.alpha)

    best_auc = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
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
            n_batches  += 1

        avg_loss = total_loss / n_batches
        avg_frac = total_frac / n_batches
        lr_now   = optimizer.param_groups[0]['lr']
        elapsed  = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.6f}  "
              f"frac_active={avg_frac:.3f}  lr={lr_now:.2e}  [{elapsed:.1f}s]")

        # ── Periodic validation ────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            # Centroid is the mean of BENIGN training embeddings only (Eq. 8)
            benign_idx = (y_train == 0)
            z_train, _ = get_features(
                model, make_eval_loader(X_train[benign_idx], y_train[benign_idx],
                                        args.batch_size), device)
            centroid    = get_centroid(z_train)

            z_val, y_val_t = get_features(model, val_loader, device)
            # anomaly score = 1 - cosine_similarity (high = anomaly)
            sim_val    = cosine_scores_to_centroid(z_val, centroid)
            anom_val   = -sim_val   # negate: high = attack

            thr, f1_val, prec, rec = best_f1_threshold(y_val.astype(int), anom_val)
            auc_roc = __import__('sklearn.metrics', fromlist=['roc_auc_score']
                                 ).roc_auc_score(y_val, anom_val)
            print(f"  [Val]  AUC-ROC={auc_roc:.4f}  F1={f1_val:.4f}  "
                  f"P={prec:.4f}  R={rec:.4f}  thr={thr:.4f}")

            if auc_roc > best_auc:
                best_auc = auc_roc
                save_checkpoint(
                    os.path.join(args.ckpt_dir, 'best.pt'),
                    model, optimizer, epoch,
                    {'auc_roc': auc_roc, 'f1': f1_val,
                     'threshold': float(thr),
                     'centroid': centroid.numpy().tolist()}
                )
                print(f"  [BEST] Checkpoint saved (AUC-ROC={best_auc:.4f})")

    print(f"\nTraining done. Best val AUC-ROC: {best_auc:.4f}")
    print(f"Checkpoint: {args.ckpt_dir}/best.pt")
    print("Run eval.py to evaluate on the test set.")


if __name__ == '__main__':
    main()
