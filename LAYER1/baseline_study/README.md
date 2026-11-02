# Baseline study - training code

Training scripts of the extended Layer 1 baselines (Protocol A, 14 leave-one-attack-out folds),
written and run by the teammates who produced the baseline checkpoints. Kept here as the record
of how those checkpoints were trained; they are scored and compared by
`LAYER1/experiments/layer1_eval/` (notebook `LAYER1/experiments/layer1_results.ipynb`).

| Script | Model | Checkpoint folder (`UTS_BASELINE_ART`) |
|---|---|---|
| `train_xgboost_only.py` | XGBoost on 78 features | `xgboost_only/` (same files as the paper's run) |
| `train_lgbm_only.py` | LightGBM | `lgbm/` |
| `train_random_forest.py` | Random Forest, 200 trees | `random_forest/` |
| `train_knn_only.py` | k-NN (PyTorch), k = 5, 50K training flows | `knn/` |
| `train_svm_only.py` | Linear SVM | `svm_only/` |
| `train_ae_only.py` | Benign autoencoder | `ae_only/` |
| `train_vae_only.py` | Benign VAE, latent 32 | `vae_only/` |
| `train_svdd_only.py`, `train_deep_svdd.py` | Mixture Deep SVDD (the same model) | `svdd_only/`, `deep_svdd/` |
| `train_clad.py` | CLAD | `clad/` |
| `train_ae_xgboost.py`, `train_svdd_xgboost.py`, `train_vae_xgboost.py` | XGBoost on features + encoder signals | `ae_xgboost/`, `svdd_xgboost/`, `vae_xgboost/` |
| `train_optiforest_only.py` | OptiForest (scored with the authors' code, `UTS_OPTIFOREST`) | `optiforest/` |

Notes for reuse:

- The scripts expect the layout of the original project: `config/labels.json`, `common/`
  (XGBoost configs, also in `LAYER1/`), `models/` (autoencoder, SVDD, CLAD as in `LAYER1/`, plus a
  VAE module and `models/clad/util/metrics.threshold_by_fpr`, not included here) and
  `data/CICIDS2018/by_label/<fold>/` (= `03_splits_v2/by_label`, override with `SPLITS_DIR`).
- Their own thresholds differ by model (mean of per-fold benign-train quantiles at 0.5% for the
  XGBoost family, LightGBM, k-NN; per-fold quantile for Random Forest and SVM; F1-max on
  `val_binary` for CLAD). The reported comparison re-thresholds every model with one rule; see
  `LAYER1/experiments/README.md`.
- `train_vae_xgboost.py` uses its own XGBoost settings (`max_depth=5`, `min_child_weight=10`,
  `colsample_bytree=1.0`, `reg_alpha=1.0`); the other XGBoost models share `common/xgb_config_v5_mono.py`.
