# CLAD on CIC-IDS-2018

Implementation of the **CLAD** (Contrastive Learning for Anomaly Detection) method from the CLOSR paper
([Wilkie et al., IEEE TNSM 2026](https://github.com/jackwilkie/CLOSR))
adapted for the pre-split CIC-IDS-2018 dataset in this repo.

> CLOSR multi-class is not used here because the dataset only has binary labels (0=benign, 1=attack).
> CLAD is the appropriate variant for binary anomaly detection.

## How it works

1. **Train** a contrastive MLP encoder **on benign-only traffic** (`train_isolation_benign.csv`).
   The CLAD loss pulls benign embeddings toward a compact cluster on a unit hypersphere.
2. At inference, compute the **cosine similarity** of each sample to the benign centroid.
   Low similarity → far from benign cluster → anomaly.
3. **Threshold** is selected by maximising F1 on `val_binary.csv` (PR-curve method).
4. Final numbers reported on `test_binary.csv` only.

## File structure

```
closr/
  train.py              # Training script
  eval.py               # Evaluation script (loads checkpoint, reports test metrics)
  model.py              # ContrastiveMLP architecture
  data/
    dataset.py          # CSV loaders, ZNormaliser, DataLoader helpers
  losses/
    clad_loss.py        # CLADLoss (positive compactness + negative margin)
  util/
    distance.py         # Centroid computation, cosine scoring
    metrics.py          # AUC-ROC, AUC-PR, FPR@95, F1, threshold search
    schedules.py        # Warmup + cosine LR schedule
    checkpoint.py       # save / load model checkpoints
  checkpoints/
    best.pt             # Best model (created after training)
    eval_results.png    # ROC/PR/score-dist plots (created after eval)
```

## Commands (run from repo root)

```bash
# Full training — 200 epochs, data/ split
python -m closr.train --data_dir data --epochs 200

# Use the larger 2M split
python -m closr.train --data_dir 2M_Benign_500K_attack --epochs 200

# Quick smoke test (2 epochs)
python -m closr.train --data_dir data --epochs 2 --eval_every 1

# Evaluate best checkpoint on test set
python -m closr.eval --data_dir data

# Evaluate with specific checkpoint path
python -m closr.eval --data_dir data --ckpt closr/checkpoints/best.pt
```

## Key hyperparameters

| Param | Default | Paper value |
|---|---|---|
| `hidden_dim` | 1024 | 1024 |
| `n_layers` | 4 | 4 |
| `d_out` | 8 | 8 |
| `peak_lr` | 4.4e-6 | 4.4e-6 |
| `margin` | 1.0 | 1.0 |
| `alpha` | 0.5 | 0.5 |
| `epochs` | 200 | 200 |
| `batch_size` | 4096 | 4096 |

## Score convention

`anomaly_score = -cosine_similarity(embedding, benign_centroid)`

High score = attack (consistent with sklearn `precision_recall_curve` convention,
and consistent with how OptIForest/SVDD scores are negated in this repo).
