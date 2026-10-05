# Extended Layer 1 baselines at a common false-positive budget

Re-evaluates every Layer 1 baseline under one threshold rule, so that known-attack
detection and zero-shot accuracy are compared at the same false-positive rate.

**Threshold rule (all models).** For each leave-one-attack-out fold, the threshold is the
`1 - alpha` quantile of the model's scores on the 560K benign training flows
(`np.quantile(..., method="higher")`); a flow is an alert if `score >= threshold`.
Validation and test data never enter the threshold. Reported at `alpha = 0.5%`
(quantile 0.995, the budget of the baseline study) and `alpha = 0.4%` (the paper's
operating point), plus a sweep and the threshold-free pAUC over FPR in [0, 3%].

Same rule and metric code as `../rq1_rq2_layer1/04_layer1_tables.py`; the paper's six
models are read from that study's `sweep_all_scores.npz` and listed alongside.

## Inputs

| What | Where (environment variable, default) |
|---|---|
| Protocol A splits: `train_benign_only.csv`, `by_label/<fold>/{val_binary,test_zeroshot}.csv` | `UTS_BENCH` → `zeroshot_benchmark/03_splits_v2` |
| Baseline checkpoints `<model>/<fold>/checkpoint/...`, `<model>/checkpoint/...`, `<model>/metrics.json` | `UTS_BASELINE_ART` → `<workspace>/baselines_friends/artifacts` |
| Layer 1 model classes (`models/autoencoder.py`, `models/svdd.py`, `models/clad/`) | `UTS_LAYER1_CODE` → `zeroshot_benchmark/04_training_v2` (repo: `LAYER1/`) |
| Paper models' scores `sweep_all_scores.npz` | `UTS_RESULTS/rq1_rq2_layer1` (made by `../rq1_rq2_layer1/01-03`) |

## Run (in order)

```bash
python 01_score_baselines.py            # scores of every available model -> scores_<model>.npz
python 02_verify_reproduction.py        # each trainer's own threshold vs its metrics.json -> verify.md
python 03_baseline_tables.py [--alphas 0.005 0.004]
                                        # common-budget tables; per alpha: table_alpha<a>.csv,
                                        # thresholds_alpha<a>.csv (every model x fold threshold),
                                        # plus baseline_tables_<tag>.md and manifest_<tag>.json
python 04_baseline_latency.py [model]   # per-flow latency, CPU, 20 warm-up + 300 calls -> baseline_latency.json
                                        # (with model names: measure only those and merge into the file)
python 03_baseline_tables.py --alphas 0.001 0.002 0.004 0.008 0.01 0.015 0.02 0.03
python 05_plot_zs_vs_alpha.py           # figure: ZS-Acc vs alpha for five models -> zs_vs_alpha.pdf/.png
```

`01` accepts model names (`python 01_score_baselines.py lgbm knn`) and skips models whose
checkpoints are missing. Outputs go to `UTS_RESULTS/rq1_rq2_baselines/`.
On a 16 GB machine set `OPENBLAS_NUM_THREADS=2`.

## Models (`baselines.py`)

| Key | Model | Family | Score |
|---|---|---|---|
| `xgboost_only` | XGBoost on 78 features | Supervised | P(attack) |
| `lgbm` | LightGBM | Supervised | P(attack) |
| `random_forest` | Random Forest, 200 trees | Supervised | P(attack) |
| `knn` | k-NN, k = 5, distance-weighted, 50K training flows | Supervised | P(attack) |
| `svm_only` | Linear SVM | Supervised | signed margin |
| `ae_only` | Benign autoencoder | One-class | reconstruction error |
| `svdd_only` | Mixture Deep SVDD | One-class | distance to nearest centre |
| `vae_only` | Benign VAE, latent 32 | One-class | reconstruction MSE (+ weighted KL, see `VAE_KL_WEIGHT`) |
| `clad` | CLAD | Contrastive | −cos(z, benign centroid) |
| `ae_xgboost`, `svdd_xgboost`, `vae_xgboost` | XGBoost on features + encoder signals | Hybrid | P(attack) |

Two model classes are not in the training code shipped with the checkpoints and are rebuilt
in `baselines.py` from what the checkpoints store: the k-NN (`PyTorchKNNClassifier`, pickled
under `__main__`: 50K scaled training flows, k = 5, inverse-distance vote) and the VAE
(architecture from the parameter names and shapes of `vae_best.pt` and `vae_params` in
`metrics.json`). Step 2 confirms both against the trainer's reported metrics.
OptiForest needs its own library and is not re-scored.

`ae_xgboost`, `svdd_xgboost`, `xgboost_only`, `clad`, `ae_only` and `svdd_only` re-train
models the paper already reports; the tables list them as re-training variance.

**Verification.** `02` must report `yes` for every model before the tables are used: it
means the checkpoint, its scaler and its feature construction reproduce the trainer's
own reported metrics, so the common-budget numbers describe the same trained models.

## Reproducing a reported number

`thresholds_alpha<a>.csv` stores the exact threshold (`repr` of the float) of every model and
fold with its realized validation FPR, recall and zero-shot accuracy; `manifest_<tag>.json`
records the code version and the MD5 of every score file used. Re-running `03` with the same
`--alphas` on the same score files reproduces the tables to the last digit.
