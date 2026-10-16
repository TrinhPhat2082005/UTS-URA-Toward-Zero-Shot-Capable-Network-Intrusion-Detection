"""The configuration behind the reported Layer 1 comparison (slides of 6 Oct 2026, paper tables).

One place that fixes, for every row of the final table and chart:
  - which checkpoint is used: "paper" = the paper's own run (rq1_rq2_layer1/sweep_all_scores.npz,
    checkpoints under UTS_LAYER1_ART) or "baseline-study" = the teammates' run (scores_<key>.npz,
    checkpoints under UTS_BASELINE_ART);
  - the threshold rule and the false-positive budgets.
06_final_results.py reads this file; the slide and figure scripts read 06's outputs.

Choice recorded on 2026-10-05: the AE models from the paper's run, the SVDD-XGBoost from the
baseline study's run (re-trained encoders differ between runs, see README), every other baseline
from the baseline study. XGBoost-only is byte-identical in both runs.

Revised 2026-10-16: the table separates the two SVDD one-class models, which earlier versions
of this file conflated.

  - Deep SVDD (key deep_svdd) is the published one-class detector: one encoder, one centre,
    trained on benign traffic alone. It is a baseline, not part of this work.
  - Mixture Deep SVDD (key svdd_only) is this work's variant: K-means partitions the benign
    traffic and one sphere is fitted per cluster, so that the per-sphere distances and their
    minimum can augment the XGBoost gate. As a detector on its own it is weak, which is why it
    appears here next to its hybrid form rather than only inside SVDD-XGBoost.

Both are scored by 01 like every other baseline, so both rows are reproduced from
scores_<key>.npz rather than quoted from a metrics file. OptiForest stays out of the table and
is recorded in the README: it needs its own library and is not re-scored under this rule.
"""
RULE = "meanfold"                    # 03_baseline_tables.py --rule: mean of the 14 fold thresholds
ALPHA_TABLE = 0.005                  # budget of the main table (quantile 0.995)
ALPHA_GRID = (0.001, 0.002, 0.003, 0.004, 0.005)   # budgets on the zero-shot curve

P, B = "paper", "baseline-study"
# role: (display name, family, key in the tables, source); order = table order
ROWS = [
    ("xgb",      "XGBoost-only",        "Supervised",  "xgboost_only",      P),
    ("lgbm",     "LightGBM",            "Supervised",  "lgbm",              B),
    ("rf",       "Random Forest",       "Supervised",  "random_forest",     B),
    ("knn",      "KNN",                 "Supervised",  "knn",               B),
    ("svm",      "SVM (linear)",        "Supervised",  "svm_only",          B),
    ("ae_only",  "AE-only",             "One-class",   "ae_only",           P),
    ("vae_only", "VAE-only",            "One-class",   "vae_only",          B),
    ("mix_svdd", "Mixture Deep SVDD",   "One-class",   "svdd_only",         B),
    ("deep_svdd", "Deep SVDD",          "One-class",   "deep_svdd",         B),
    ("clad",     "CLAD",                "Contrastive", "clad",              B),
    ("vae_xgb",  "VAE-XGBoost",         "Hybrid",      "vae_xgboost",       B),
    ("ae_xgb",   "AE-XGBoost (ours)",   "Hybrid",      "ae_xgboost_mono",   P),
    ("svdd_xgb", "SVDD-XGBoost (ours)", "Hybrid",      "svdd_xgboost",      B),
]
CURVE = ["xgb", "clad", "vae_xgb", "ae_xgb", "svdd_xgb"]   # models on the zero-shot curve
OURS = {"ae_xgb", "svdd_xgb"}

# Latency of each role: decision only, measured by 04_baseline_latency.py on the baseline study's
# fold-Bot checkpoint of the same architecture (one run, one machine for all rows).
LATENCY_KEY = {"xgb": "xgboost_only", "lgbm": "lgbm", "rf": "random_forest", "knn": "knn", "svm": "svm_only",
               "ae_only": "ae_only", "vae_only": "vae_only", "mix_svdd": "svdd_only",
               "deep_svdd": "deep_svdd", "clad": "clad", "vae_xgb": "vae_xgboost",
               "ae_xgb": "ae_xgboost", "svdd_xgb": "svdd_xgboost"}
