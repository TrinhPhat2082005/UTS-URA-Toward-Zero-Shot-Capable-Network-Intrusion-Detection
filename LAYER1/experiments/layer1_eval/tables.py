"""Bang Layer 1 tren slide (RQ1/RQ2): moi model mot dong, cung ngan sach FPR, cung quy tac nguong.

Cau hinh cua bang (ROWS):
  - checkpoint cua moi dong: "paper" = run cua paper (UTS_LAYER1_ART), "baseline-study" = run cua teammate
    (UTS_BASELINE_ART). Chon ngay 2026-10-05: hai model AE lay tu run cua paper, SVDD-XGBoost lay tu baseline
    study (encoder train lai khac nhau giua hai run, xem extras.checkpoint_runs), cac baseline con lai lay tu
    baseline study; XGBoost-only giong het nhau o hai run.
  - Deep SVDD (deep_svdd) la detector one-class da cong bo (mot encoder, mot tam). Mixture Deep SVDD
    (svdd_only) la bien the cua paper: moi cum K-means cua benign mot sphere, de khoang cach toi cac sphere
    lam feature cho XGBoost; dung mot minh thi yeu, nen dat canh dang hybrid cua no.
  - OptiForest cham bang code cua tac gia (UTS_OPTIFOREST), xem scores._compute_optiforest.

Trong notebook: results[role] = tables.run(role) cho moi dong, roi tables.write_final(results) ghi OUT_FINAL/:
    table_main.csv       moi model mot dong o ALPHA_TABLE (+ pAUC, latency)        -> bang tren slide
    curve.csv            ZS-Acc va FPR thuc te cua cac model CURVE o ALPHA_GRID    -> hinh ZS-Acc theo alpha
    thresholds_main.csv  nguong ap dung o tung fold tai ALPHA_TABLE, kem FPR / recall / ZS-Acc cua fold
    final_results.md     hai bang tren, dang doc duoc
    manifest.json        phien ban code va thu vien, quy tac, MD5 cua moi checkpoint va file diem
"""
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime

import pandas as pd

from . import metrics, paths, scorers, scores

RULE = "meanfold"                                    # trung binh 14 nguong cua cac fold
ALPHA_TABLE = 0.005                                  # ngan sach FPR cua bang chinh (quantile 0.995)
ALPHA_GRID = (0.001, 0.002, 0.003, 0.004, 0.005)     # cac ngan sach tren duong ZS-Acc
P, B = scores.PAPER, scores.BASE

# role: ten hien thi, family, khoa model, checkpoint; thu tu = thu tu dong trong bang
ROWS = [
    ("xgb",        "XGBoost-only",        "Supervised",  "xgboost_only",    P),
    ("lgbm",       "LightGBM",            "Supervised",  "lgbm",            B),
    ("rf",         "Random Forest",       "Supervised",  "random_forest",   B),
    ("knn",        "KNN",                 "Supervised",  "knn",             B),
    ("svm",        "SVM (linear)",        "Supervised",  "svm_only",        B),
    ("ae_only",    "AE-only",             "One-class",   "ae_only",         P),
    ("vae_only",   "VAE-only",            "One-class",   "vae_only",        B),
    ("mix_svdd",   "Mixture Deep SVDD",   "One-class",   "svdd_only",       B),
    ("deep_svdd",  "Deep SVDD",           "One-class",   "deep_svdd",       B),
    ("optiforest", "OptiForest",          "One-class",   "optiforest",      B),
    ("clad",       "CLAD",                "Contrastive", "clad",            B),
    ("vae_xgb",    "VAE-XGBoost",         "Hybrid",      "vae_xgboost",     B),
    ("ae_xgb",     "AE-XGBoost (ours)",   "Hybrid",      "ae_xgboost_mono", P),
    ("svdd_xgb",   "SVDD-XGBoost (ours)", "Hybrid",      "svdd_xgboost",    B),
]
ROW = {r[0]: r for r in ROWS}
CURVE = ["xgb", "clad", "vae_xgb", "ae_xgb", "svdd_xgb"]   # cac model tren hinh ZS-Acc theo alpha

# Latency cua moi dong: chi phan quyet dinh, do boi extras.latency tren checkpoint fold Bot cua baseline study
# cung kien truc (mot lan do, mot may cho moi dong).
LATENCY_KEY = {"xgb": "xgboost_only", "lgbm": "lgbm", "rf": "random_forest", "knn": "knn", "svm": "svm_only",
               "ae_only": "ae_only", "vae_only": "vae_only", "mix_svdd": "svdd_only", "deep_svdd": "deep_svdd",
               "optiforest": "optiforest", "clad": "clad", "vae_xgb": "vae_xgboost", "ae_xgb": "ae_xgboost",
               "svdd_xgb": "svdd_xgboost"}
METRICS = ("acc", "precision", "recall", "f1", "fpr", "zs_acc")


def pct(a):
    return f"{100 * a:g}"


def evaluate_model(key, source, recompute=False):
    """Moi chi so cua mot checkpoint: bang o ALPHA_TABLE, duong ZS-Acc o ALPHA_GRID, pAUC, nguong tung fold."""
    S = scores.get(key, source, recompute)
    table, per_fold, thr = metrics.evaluate(S, key, ALPHA_TABLE, RULE)
    return {"key": key, "source": source, "table": table, "per_fold": per_fold, "thresholds": thr,
            "curve": {a: metrics.evaluate(S, key, a, RULE)[0] for a in ALPHA_GRID},
            "pauc": metrics.pauc(S, key), "n_benign_train": {f: len(S[f"{key}|{f}|tb"]) for f in thr}}


def run(role, recompute=False):
    """Mot dong cua bang: doc (hoac cham lai) diem, tinh chi so, in tom tat."""
    _, name, family, key, source = ROW[role]
    r = {"role": role, "model": name, "family": family, **evaluate_model(key, source, recompute)}
    t = r["table"]
    print(f"{name} [{source}]  alpha {pct(ALPHA_TABLE)}% {RULE}:  Acc {t['acc']:.4f}  Precision {t['precision']:.4f}  "
          f"Recall {t['recall']:.4f}  F1 {t['f1']:.4f}  FPR {100 * t['fpr']:.2f}%  ZS-Acc {t['zs_acc']:.4f}  "
          f"pAUC[0,3%] {r['pauc']:.4f}")
    return r


def fold_table(r):
    """Tung fold cua mot ket qua: nguong ap dung, phan vi rieng cua fold, chi so o nguong do."""
    rows = {f: {"threshold": t, "fold_quantile": q, **r["per_fold"][f]} for f, (t, q) in r["thresholds"].items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


ENCODER_DIR = {"ae_xgboost": "ae_only", "ae_xgboost_mono": "ae_only", "svdd_xgboost": "svdd_only",
               "svdd_xgboost_mono": "svdd_only", "vae_xgboost": "vae_only"}


def _checkpoint_md5(key, source):
    """MD5 cua moi file checkpoint dung de cham dong nay (checkpoint khong nam trong git)."""
    root = scorers.ROOT[source]
    dirs = [scorers.MODELS[key].dir] + ([ENCODER_DIR[key]] if key in ENCODER_DIR else [])
    files = sorted(p for d in dirs for p in (root / d).rglob("*")
                   if p.is_file() and p.suffix in (".pt", ".pkl", ".npy", ".json"))
    return {p.relative_to(root).as_posix(): _md5(p) for p in files}


def _code_version():
    try:
        return subprocess.run(["git", "-C", str(paths.LAYER1), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "not a git checkout"


def _versions():
    out = {"python": platform.python_version()}
    for mod in ("numpy", "sklearn", "xgboost", "torch", "lightgbm", "joblib"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:        # thieu thu vien, hoac thieu dependency cua no
            out[mod] = f"unavailable ({type(e).__name__})"
    return out


def write_final(results, out=paths.OUT_FINAL):
    """Ghi bang cuoi cung tu results = {role: run(role)} cua du 14 dong."""
    missing = [role for role in ROW if role not in results]
    assert not missing, f"chua chay cac dong: {missing}"
    out = paths.ensure(out)
    lat_file = paths.OUT_BASE / "baseline_latency.json"
    lat = json.loads(lat_file.read_text(encoding="utf-8")) if lat_file.exists() else {}

    main_rows, curve_rows, thr_rows = [], [], []
    for role, name, family, key, source in ROWS:
        r, d = results[role], lat.get(f"{LATENCY_KEY[role]}/decision")
        main_rows.append({"role": role, "model": name, "family": family, "key": key, "source": source,
                          **{k: round(r["table"][k], 6) for k in METRICS}, "pauc": round(r["pauc"], 6),
                          "latency_ms": round(d["mean_ms"], 3) if d else ""})
        for f, (t, q) in r["thresholds"].items():
            m = r["per_fold"][f]
            thr_rows.append({"alpha": ALPHA_TABLE, "rule": RULE, "key": key, "model": name, "source": source,
                             "fold": f, "threshold": repr(t), "fold_quantile": repr(q),
                             "n_benign_train": r["n_benign_train"][f], "val_fpr": round(m["fpr"], 6),
                             "val_recall": round(m["recall"], 6), "zs_acc": round(m["zs_acc"], 6)})
    for role in CURVE:
        r = results[role]
        for a in ALPHA_GRID:
            curve_rows.append({"role": role, "model": r["model"], "key": r["key"], "source": r["source"], "alpha": a,
                               "zs_acc": round(r["curve"][a]["zs_acc"], 6), "fpr": round(r["curve"][a]["fpr"], 6),
                               "pauc": round(r["pauc"], 6)})
    _write_csv(out / "table_main.csv", main_rows)
    _write_csv(out / "curve.csv", curve_rows)
    _write_csv(out / "thresholds_main.csv", thr_rows)

    md = [f"## Layer 1 models at alpha = {pct(ALPHA_TABLE)}% ({RULE})\n",
          "| Model | Family | Checkpoint | Acc | Precision | Recall | F1 | FPR | ZS-Acc | pAUC[0,3%] | ms/flow |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in main_rows:
        md.append(f"| {r['model']} | {r['family']} | {r['source']} | "
                  + " | ".join(f"{r[k]:.4f}" for k in ("acc", "precision", "recall", "f1"))
                  + f" | {100 * r['fpr']:.2f}% | {r['zs_acc']:.4f} | {r['pauc']:.4f} | {r['latency_ms']} |")
    md += [f"\n## Zero-shot accuracy vs alpha ({RULE}); realized val FPR in brackets\n",
           "| Model | " + " | ".join(f"{pct(a)}%" for a in ALPHA_GRID) + " |",
           "|---|" + "---|" * len(ALPHA_GRID)]
    for role in CURVE:
        pts = [c for c in curve_rows if c["role"] == role]
        md.append(f"| {pts[0]['model']} | "
                  + " | ".join(f"{c['zs_acc']:.3f} ({100 * c['fpr']:.2f}%)" for c in pts) + " |")
    (out / "final_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("hashing checkpoints ...", flush=True)
    ckpts = {role: _checkpoint_md5(key, source) for role, _, _, key, source in ROWS}
    score_files = sorted({scores.cache_file(k, s) for _, _, _, k, s in ROWS})
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "code_version": _code_version(),
                "libraries": _versions(), "threshold_rule": RULE, "alpha_table": ALPHA_TABLE,
                "alpha_grid": list(ALPHA_GRID), "folds": list(results[ROWS[0][0]]["thresholds"]),
                "selection": [{"role": ro, "model": n, "key": k, "source": s} for ro, n, _, k, s in ROWS],
                "checkpoint_roots": {k: str(v) for k, v in scorers.ROOT.items()}, "checkpoints_md5": ckpts,
                "inputs_md5": {p.name: _md5(p) for p in score_files + [lat_file] if p.exists()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("\n".join(md))
    print(f"-> {paths.rel(out)}")
    return pd.DataFrame(main_rows).set_index("model")
