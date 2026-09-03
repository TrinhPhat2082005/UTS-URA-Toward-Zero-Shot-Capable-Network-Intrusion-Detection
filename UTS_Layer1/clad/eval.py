"""
Evaluate a trained CLAD checkpoint on the test set.

Usage:
    python closr/eval.py                          # uses data/ + closr/checkpoints/best.pt
    python closr/eval.py --data_dir 2M_Benign_500K_attack
    python closr/eval.py --ckpt closr/checkpoints/best.pt --data_dir data

Prints:
  - Val metrics (as reference, threshold is tuned here)
  - Test metrics with the val-tuned threshold
  - Saves a comparison plot: closr/checkpoints/eval_results.png
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from closr.data.dataset    import load_splits, make_eval_loader
from closr.model           import build_model
from closr.util.checkpoint import load_checkpoint
from closr.util.distance   import get_features, get_centroid, cosine_scores_to_centroid
from closr.util.metrics    import best_f1_threshold, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default='data')
    p.add_argument('--ckpt',       default='closr/checkpoints/best.pt')
    p.add_argument('--train_file', default='train_xgboost_binary.csv')
    p.add_argument('--batch_size', type=int, default=4096)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--n_layers',   type=int, default=4)
    p.add_argument('--d_out',      type=int, default=32)
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args  = parse_args()
    device = torch.device(args.device)

    # ── Data ──────────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test, _ = load_splits(
        args.data_dir, train_file=args.train_file)
    # Centroid is computed from benign training embeddings only (Eq. 8)
    benign_idx   = (y_train == 0)
    train_loader = make_eval_loader(X_train[benign_idx], y_train[benign_idx], args.batch_size)
    val_loader   = make_eval_loader(X_val,   y_val,   args.batch_size)
    test_loader  = make_eval_loader(X_test,  y_test,  args.batch_size)

    # ── Model ─────────────────────────────────────────────────────────
    d_in  = X_train.shape[1]
    model = build_model(d_in=d_in, hidden_dim=args.hidden_dim,
                        n_layers=args.n_layers, d_out=args.d_out).to(device)
    epoch, saved = load_checkpoint(args.ckpt, model, device=args.device)
    print(f"Loaded checkpoint from epoch {epoch}  |  saved metrics: {saved}")

    # ── Centroid from benign train ─────────────────────────────────────
    # If centroid was saved in checkpoint, use it; otherwise recompute
    if 'centroid' in saved:
        centroid = torch.tensor(saved['centroid'], dtype=torch.float32)
        print("Centroid loaded from checkpoint.")
    else:
        z_train, _ = get_features(model, train_loader, device)
        centroid    = get_centroid(z_train)
        print("Centroid recomputed from training set.")

    # ── Scores ────────────────────────────────────────────────────────
    z_val,  _        = get_features(model, val_loader,  device)
    z_test, _        = get_features(model, test_loader, device)

    # anomaly score = negative cosine similarity (high = attack)
    anom_val  = -cosine_scores_to_centroid(z_val,  centroid)
    anom_test = -cosine_scores_to_centroid(z_test, centroid)

    # ── Threshold on val ──────────────────────────────────────────────
    thr, f1_val, prec_val, rec_val = best_f1_threshold(y_val, anom_val)
    print(f"\nVal threshold (PR-curve F1): {thr:.6f}")
    val_res  = evaluate(y_val,  anom_val,  thr,  label="CLAD — Validation")
    test_res = evaluate(y_test, anom_test, thr,  label="CLAD — Test")

    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. ROC curves
    for split, y, scores, style in [
        ('Val',  y_val,  anom_val,  'b-'),
        ('Test', y_test, anom_test, 'r-'),
    ]:
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        axes[0].plot(fpr, tpr, style, linewidth=2, label=f"{split} AUC={auc:.4f}")
    axes[0].plot([0,1],[0,1],'k--', alpha=0.4)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC Curve — CLAD')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # 2. PR curves
    for split, y, scores, style in [
        ('Val',  y_val,  anom_val,  'b-'),
        ('Test', y_test, anom_test, 'r-'),
    ]:
        prec, rec, _ = precision_recall_curve(y, scores)
        axes[1].plot(rec, prec, style, linewidth=2, label=split)
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve — CLAD')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # 3. Anomaly score distributions (test set)
    benign_scores = anom_test[y_test == 0]
    attack_scores = anom_test[y_test == 1]
    axes[2].hist(benign_scores, bins=80, alpha=0.6, color='green',
                 density=True, label='Benign')
    axes[2].hist(attack_scores, bins=80, alpha=0.6, color='red',
                 density=True, label='Attack')
    axes[2].axvline(thr, color='navy', linestyle='--',
                    label=f'Threshold={thr:.4f}')
    axes[2].set_xlabel('Anomaly Score (−cosine sim)')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Score Distribution — Test Set')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('CLAD — CIC-IDS-2018', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(args.ckpt), 'eval_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved: {out_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    print(f"{'Metric':<15} {'Val':>10} {'Test':>10}")
    print("-"*35)
    for k in ('auc_roc', 'auc_pr', 'fpr95', 'f1'):
        print(f"{k:<15} {val_res[k]:>10.4f} {test_res[k]:>10.4f}")
    print("="*55)


if __name__ == '__main__':
    main()
