"""Step 6 - the reported Layer 1 comparison, exactly as on the slides / in the paper.

Applies final_selection.py to the tables of step 3 and writes OUT_BASE/final/:
    table_main.csv       one row per model at ALPHA_TABLE (+ pAUC, decision latency)  -> slide 13
    curve.csv            ZS-Acc and realized FPR of the CURVE models at ALPHA_GRID     -> slide 14, figure
    thresholds_main.csv  the threshold applied in every fold at ALPHA_TABLE
    final_results.md     the same tables, readable
    manifest.json        code version, library versions, rule, budgets, and the MD5 of every
                         checkpoint and score file behind each row

Prerequisites (see README): rq1_rq2_layer1/01-03 (paper models), then 01, 02, 04 of this folder, and
    python 03_baseline_tables.py --rule meanfold --alphas 0.001 0.002 0.003 0.004 0.005
"""
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import final_selection as FS  # noqa: E402
from baselines import REQUIRED, folds  # noqa: E402

OUT = paths.ensure(paths.OUT_BASE / "final")


def pct(a):
    return f"{100 * a:g}"


def sfx():
    return "" if FS.RULE == "perfold" else f"_{FS.RULE}"


def read_csv(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_files(key, source):
    """Every file the scores of (key, source) were computed from."""
    if source == FS.Q:                                    # quoted row: the trainer's own outputs
        d = paths.BASE_ART / key
        return sorted(p for p in d.rglob("*")
                      if p.is_file() and p.suffix in (".pt", ".pkl", ".npy", ".json"))
    if source == FS.B:
        files = [paths.BASE_ART / r.format(fold=f) for r in REQUIRED[key]
                 for f in (folds() if "{fold}" in r else [None])]
        return sorted(set(files))
    root = paths.L1_ART                                   # the paper's run
    enc = {"ae_xgboost_mono": ["ae_only"], "svdd_xgboost_mono": ["svdd_only"]}.get(key, [])
    files = []
    for d in [key] + enc:
        if (root / d / "checkpoint").exists():            # one-class encoder: one checkpoint for all folds
            files += [p for p in (root / d / "checkpoint").rglob("*")
                      if p.is_file() and p.suffix in (".pt", ".pkl", ".npy", ".json")]
        else:
            files += [p for f in folds() for p in (root / d / f / "checkpoint").glob("*") if p.is_file()]
    return sorted(files)


def versions():
    out = {"python": platform.python_version()}
    for mod in ("numpy", "sklearn", "xgboost", "torch", "lightgbm", "joblib"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:        # missing, or a missing dependency of it
            out[mod] = f"unavailable ({type(e).__name__})"
    return out


def code_version():
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "not a git checkout"


def main():
    grid_tag = "_".join(f"alpha{pct(a)}" for a in FS.ALPHA_GRID)
    grid_csv = paths.OUT_BASE / f"table_{grid_tag}{sfx()}.csv"
    thr_csv = paths.OUT_BASE / f"thresholds_alpha{pct(FS.ALPHA_TABLE)}{sfx()}.csv"
    lat_json = paths.OUT_BASE / "baseline_latency.json"
    grid = read_csv(grid_csv)
    lat = json.loads(lat_json.read_text(encoding="utf-8"))

    def pick(key, source, alpha):
        hit = [r for r in grid if r["key"] == key and r["source"] == source and abs(float(r["alpha"]) - alpha) < 1e-12]
        assert len(hit) == 1, f"{key}/{source} at alpha={alpha}: {len(hit)} rows in {grid_csv.name}"
        return hit[0]

    def quoted(key):
        """A row the trainer reported itself, for a model 01 cannot re-score.

        Read from <key>/metrics.json under UTS_BASELINE_ART; that run applied the same budget
        on the benign training pool. pAUC needs the score arrays and is left empty.
        """
        s = json.loads((paths.BASE_ART / key / "metrics.json").read_text(encoding="utf-8"))["summary"]
        ka, zs = s["known_attack_macro_avg"], s["zero_shot_macro_avg"]
        return {"acc": ka["accuracy"], "precision": ka["precision"], "recall": ka["recall"],
                "f1": ka["f1"], "fpr": ka["fpr"], "zs_acc": zs["accuracy"], "pauc": ""}

    main_rows, curve_rows, md = [], [], []
    for role, name, fam, key, src in FS.ROWS:
        r = quoted(key) if src == FS.Q else pick(key, src, FS.ALPHA_TABLE)
        lk = FS.LATENCY_KEY[role]
        d = lat.get(f"{lk}/decision") if lk else None
        main_rows.append({"role": role, "model": name, "family": fam, "key": key, "source": src,
                          **{k: r[k] for k in ("acc", "precision", "recall", "f1", "fpr", "zs_acc", "pauc")},
                          "latency_ms": round(d["mean_ms"], 3) if d else ""})
    for role in FS.CURVE:
        name, key, src = next((n, k, s) for ro, n, _, k, s in FS.ROWS if ro == role)
        for a in FS.ALPHA_GRID:
            r = pick(key, src, a)
            curve_rows.append({"role": role, "model": name, "key": key, "source": src, "alpha": a,
                               "zs_acc": r["zs_acc"], "fpr": r["fpr"], "pauc": r["pauc"]})
    want = {(k, s) for _, _, _, k, s in FS.ROWS}
    thr_rows = [r for r in read_csv(thr_csv) if (r["key"], r["source"]) in want]

    write_csv(OUT / "table_main.csv", main_rows)
    write_csv(OUT / "curve.csv", curve_rows)
    write_csv(OUT / "thresholds_main.csv", thr_rows)

    md.append(f"## Layer 1 models at alpha = {pct(FS.ALPHA_TABLE)}% ({FS.RULE})\n")
    md.append("| Model | Family | Checkpoint | Acc | Precision | Recall | F1 | FPR | ZS-Acc | pAUC[0,3%] | ms/flow |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in main_rows:
        pauc = f"{float(r['pauc']):.4f}" if r["pauc"] != "" else "-"
        md.append(f"| {r['model']} | {r['family']} | {r['source']} | " + " | ".join(
            f"{float(r[k]):.4f}" for k in ("acc", "precision", "recall", "f1")) +
            f" | {100 * float(r['fpr']):.2f}% | {float(r['zs_acc']):.4f} | {pauc} | {r['latency_ms']} |")
    md.append(f"\n## Zero-shot accuracy vs alpha ({FS.RULE}); realized val FPR in brackets\n")
    md.append("| Model | " + " | ".join(f"{pct(a)}%" for a in FS.ALPHA_GRID) + " |")
    md.append("|---|" + "---|" * len(FS.ALPHA_GRID))
    for role in FS.CURVE:
        pts = [c for c in curve_rows if c["role"] == role]
        md.append(f"| {pts[0]['model']} | " + " | ".join(
            f"{float(c['zs_acc']):.3f} ({100 * float(c['fpr']):.2f}%)" for c in pts) + " |")
    (OUT / "final_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("hashing checkpoints ...", flush=True)
    ckpts = {}
    for role, _, _, key, src in FS.ROWS:
        root = paths.L1_ART if src == FS.P else paths.BASE_ART
        ckpts[role] = {str(p.relative_to(root)): md5(p) for p in checkpoint_files(key, src)}
    score_files = [paths.OUT_L1 / "sweep_all_scores.npz"] + sorted(
        paths.OUT_BASE / f"scores_{k}.npz" for _, _, _, k, s in FS.ROWS if s == FS.B)
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "code_version": code_version(),
                "libraries": versions(), "threshold_rule": FS.RULE, "alpha_table": FS.ALPHA_TABLE,
                "alpha_grid": list(FS.ALPHA_GRID), "folds": folds(),
                "selection": [{"role": ro, "model": n, "key": k, "source": s} for ro, n, _, k, s in FS.ROWS],
                "checkpoint_roots": {"paper": str(paths.L1_ART), "baseline-study": str(paths.BASE_ART)},
                "checkpoints_md5": ckpts,
                "inputs_md5": {p.name: md5(p) for p in score_files + [grid_csv, thr_csv, lat_json]}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("\n".join(md))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
