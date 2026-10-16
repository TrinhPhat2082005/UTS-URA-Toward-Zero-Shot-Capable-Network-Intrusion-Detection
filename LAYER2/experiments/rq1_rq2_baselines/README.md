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
`metrics.json`). Step 2 confirms both against the trainer's reported metrics. The single-sphere
Deep SVDD is rebuilt the same way from `oneclass_deep_svdd.pt`, which stores the encoder weights
and the fixed centre (bias-free 78 - 256 - 128 - 64, LeakyReLU 0.1).

OptiForest is the one baseline this pipeline does not score itself, because its scorer lives in
the authors' own repository rather than in a package. Its run nevertheless used the same rule as
every row here, a threshold calibrated to a 0.5% budget on the benign training pool
(`target_fpr: 0.005` in `optiforest/metrics.json`), and it reports all 14 folds. At that budget
it reaches a known-attack F1 of 0.003 and a zero-shot accuracy of 0.001 at 4.9 s per flow, so it
is recorded here rather than given a row: the checkpoint, the scaler and the benign score arrays
are kept under `optiforest/` for anyone who clones the upstream code to re-run it.

**Two different SVDD models.** `deep_svdd` is the published One-Class Deep SVDD of Ruff et al.
(ICML 2018): one encoder, one fixed centre, trained on benign traffic alone. `svdd_only` is this
work's Mixture Deep SVDD, which partitions the benign traffic with K-means and fits one sphere
per cluster so that the per-sphere distances can augment the XGBoost gate. They are separate
rows of the table and separate checkpoint trees (`deep_svdd_paper/` and `svdd_only/`).

`ae_xgboost`, `svdd_xgboost`, `xgboost_only`, `clad`, `ae_only` and `svdd_only` re-train
models the paper already reports; the tables list them as re-training variance.

**Verification.** `02` must report `yes` for every model before the tables are used: it
means the checkpoint, its scaler and its feature construction reproduce the trainer's
own reported metrics, so the common-budget numbers describe the same trained models.

## Threshold rules

`03` takes `--rule perfold` (default: each fold uses its own threshold) or `--rule meanfold`
(the baseline study's rule: the per-fold thresholds, each computed on all 560K benign training
flows, are averaged into one threshold used in every fold). Files of the meanfold rule end in
`_meanfold`; `05_plot_zs_vs_alpha.py --rule` selects the matching table. The benign training
pool is the same 560K flows in every fold; thresholds differ per fold because each fold has its
own supervised model. Averaging leaves the realized FPR slightly above alpha for most models and
far above it for SVM and Random Forest, whose score scales differ between folds.

## Reproducing a reported number

`thresholds_alpha<a>.csv` stores the exact threshold (`repr` of the float) of every model and
fold with its realized validation FPR, recall and zero-shot accuracy; `manifest_<tag>.json`
records the code version and the MD5 of every score file used. Re-running `03` with the same
`--alphas` on the same score files reproduces the tables to the last digit.

## Reproducing the reported comparison (slides of 6 Oct 2026)

`final_selection.py` fixes everything the reported numbers depend on: which checkpoint plays
each row (AE-only and AE-XGBoost from the paper's run, SVDD-XGBoost and every other baseline from
the baseline study's run; XGBoost-only is the same file in both), the threshold rule
(`meanfold`), the table budget (alpha = 0.5%) and the curve budgets (0.1-0.5%).

```bash
# environment: conda env ai_env (python 3.11, numpy 2.4.6, scikit-learn 1.8.0, xgboost 3.2.0,
#              torch 2.5.1+cu121, lightgbm 4.7.0); OPENBLAS_NUM_THREADS=2 on a 16 GB machine
# paper checkpoints (UTS_LAYER1_ART) -> sweep_all_scores.npz
python ../rq1_rq2_layer1/01_score_val_test.py && python ../rq1_rq2_layer1/02_score_train_benign.py     && python ../rq1_rq2_layer1/03_score_oneclass.py
# baseline-study checkpoints (UTS_BASELINE_ART)
python 01_score_baselines.py
python 02_verify_reproduction.py          # every row must read "yes"
python 03_baseline_tables.py --rule meanfold --alphas 0.001 0.002 0.003 0.004 0.005
python 04_baseline_latency.py
python 06_final_results.py                # -> final/table_main.csv, curve.csv, thresholds_main.csv, manifest.json
python 05_plot_zs_vs_alpha.py             # -> final/zs_vs_alpha.pdf/.png
```

`final/manifest.json` lists the MD5 of every checkpoint file and score file behind each row,
the code version and the library versions. Checkpoints are not in git (several GB); compare
their MD5 with the manifest before rerunning. The baseline study's training code is kept in
`LAYER1/baseline_study/` of the repository.

Run-to-run variation: re-training the AE / SVDD encoders on another machine gives different
weights (GPU non-determinism; K-Means also differs between scikit-learn 1.8 and 1.9), and the
hybrids' zero-shot accuracy then moves by up to about 0.05, mostly through single folds such as
DDOS_attack_HOIC. The checkpoints listed in the manifest are the ones reported.
