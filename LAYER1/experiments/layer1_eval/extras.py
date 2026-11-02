"""Phan phu cua danh gia Layer 1 (notebook layer1_extras.ipynb).

    latency()               ms/flow cua moi model (cot ms/flow cua bang)        -> OUT_BASE/baseline_latency.json
    plot_zs_vs_alpha()      hinh ZS-Acc theo alpha tu final/curve.csv           -> OUT_FINAL/zs_vs_alpha.pdf/.png
    perfold_tables()        6 model cua paper, nguong rieng tung fold            -> OUT_L1/layer1_tables.md
    verify_reproduction()   nguong rieng cua nguoi train -> so voi metrics.json  -> OUT_BASE/verify.md
    checkpoint_runs()       cung model, run cua paper vs run cua baseline study (vi sao chon checkpoint nao)
"""
import json
import platform
import subprocess
import time

import numpy as np
import pandas as pd

from . import data, metrics, paths, scorers, scores, tables

# ------------------------------------------------------------------ latency
N_WARM, N_MEAS, LAT_FOLD = 20, 300, "Bot"


def _bench(fn, rows):
    ts = []
    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        fn(r)
        if i >= N_WARM:
            ts.append((time.perf_counter() - t0) * 1000)
    ts = np.asarray(ts)
    return {"mean_ms": float(ts.mean()), "p95_ms": float(np.percentile(ts, 95)), "n": int(len(ts))}


def latency(keys=None):
    """Mot flow moi lan goi (nhu khi trien khai): 20 warm-up + 300 lan do, time.perf_counter, 4 thread torch,
    checkpoint fold Bot cua baseline study, flow lay ngau nhien tu val_binary cua fold Bot. Do hai muc:
        decision             feature tho -> diem attack
        decision+treeshap    nhu tren, cong TreeSHAP chinh xac cua model cay cuoi cung (chi model cay)
    Chay voi CUDA_VISIBLE_DEVICES="" (dat truoc khi import torch) de do tren CPU.
    keys=None: moi model trong tables.LATENCY_KEY; danh sach khoa: chi do cac model do va gop vao file cu."""
    import shap
    import torch
    torch.set_num_threads(4)
    Xv, _, _ = data.fold_data(LAT_FOLD)
    idx = np.random.default_rng(0).choice(len(Xv), N_WARM + N_MEAS, replace=False)
    rows = [Xv[i:i + 1] for i in idx]
    out_f = paths.ensure(paths.OUT_BASE) / "baseline_latency.json"
    res = json.loads(out_f.read_text(encoding="utf-8")) if keys and out_f.exists() else {}
    res.update({"cpu": platform.processor(), "torch_threads": torch.get_num_threads(), "fold": LAT_FOLD,
                "n_warmup": N_WARM, "n_measured": N_MEAS})
    if platform.system() == "Windows":
        res["power_scheme"] = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                                             text=True).stdout.strip()
    for key in keys or list(dict.fromkeys(tables.LATENCY_KEY.values())):
        score = scorers.loader(key, scores.BASE, LAT_FOLD)
        res[f"{key}/decision"] = _bench(score, rows)
        model, prep = scorers.tree_model(key, scores.BASE, LAT_FOLD)
        if model is not None:
            if hasattr(model, "get_booster"):
                model.get_booster().set_param({"device": "cpu"})
            ex = shap.TreeExplainer(model)
            res[f"{key}/decision+treeshap"] = _bench(lambda r: (score(r), ex.shap_values(prep(r))), rows)
        for k in (f"{key}/decision", f"{key}/decision+treeshap"):
            if k in res:
                print(f"  {k:<36} {res[k]['mean_ms']:8.2f} ms  (p95 {res[k]['p95_ms']:7.2f})", flush=True)
    out_f.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"-> {paths.rel(out_f)}")
    return res


# ------------------------------------------------------------------ hinh ZS-Acc theo alpha
STYLE = {   # mau, do day net, marker; hai bien the cua paper ve dam hon
    "xgb": ("#9E9E9E", 1.5, "o"), "clad": ("#E08E2B", 1.5, "s"), "vae_xgb": ("#7B5EA7", 1.5, "D"),
    "ae_xgb": ("#1F497D", 2.4, "o"), "svdd_xgb": ("#2E75B6", 2.4, "^"),
}


def plot_zs_vs_alpha(curve_csv=paths.OUT_FINAL / "curve.csv"):
    """Hinh bao cao: final/curve.csv (tables.write_final) -> final/zs_vs_alpha.pdf/.png."""
    import matplotlib.pyplot as plt
    rows = pd.read_csv(curve_csv).to_dict("records")
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for role in tables.CURVE:
        pts = sorted((r["alpha"], r["zs_acc"], r["pauc"], r["model"]) for r in rows if r["role"] == role)
        col, lw, mk = STYLE[role]
        ax.plot([100 * p[0] for p in pts], [p[1] for p in pts], color=col, lw=lw, marker=mk, ms=5,
                label=f"{pts[0][3]} (pAUC$_{{[0,3\\%]}}$ {pts[0][2]:.3f})")
    alphas = sorted({100 * r["alpha"] for r in rows})
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:g}%" for a in alphas])
    how = "mean of the 14 fold thresholds" if tables.RULE == "meanfold" else "per fold"
    ax.set_xlabel(f"False-positive budget α (benign-train quantile, {how})")
    ax.set_ylabel("Zero-shot accuracy (macro, 14 folds)")
    ax.grid(axis="y", color="#DDDDDD", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(curve_csv.parent / f"zs_vs_alpha.{ext}", dpi=300)
    print(f"-> {paths.rel(curve_csv.parent / 'zs_vs_alpha.pdf')}, .png")
    return fig


# ------------------------------------------------------------------ bang nguong rieng tung fold (6 model cua paper)
PAPER_VARS = ["xgboost_only", "ae_only", "svdd_only", "clad", "ae_xgboost_mono", "svdd_xgboost_mono"]
PAPER_NAME = {"xgboost_only": "XGBoost-only", "ae_only": "AE-only", "svdd_only": "SVDD-only", "clad": "CLAD",
              "ae_xgboost_mono": "AE-XGBoost (ours)", "svdd_xgboost_mono": "SVDD-XGBoost (ours)"}
PERFOLD_ALPHAS = [0.001, 0.002, 0.004, 0.008, 0.010, 0.015, 0.020, 0.030]


def perfold_tables():
    """Bang cua 6 model cua paper voi quy tac "perfold" (tau_f rieng tung fold) -> OUT_L1/layer1_tables.md.
    T-A ZS-Acc theo alpha + pAUC; T-B known-attack o alpha 0.4% / 1.5%; T-C nguong F1-max cua code train (chi de
    tham khao, khong phai nguong cua paper); T-D ZS-Acc tung loai bi giu lai."""
    S = {}
    for v in PAPER_VARS:
        S.update(scores.get(v, scores.PAPER))
    folds = metrics.folds_of(S)
    out = []
    P = out.append

    def macro(v, a):
        return metrics.evaluate(S, v, a, "perfold")[0]

    P("## T-A. Zero-shot accuracy (macro over 14 rounds) vs. false-positive budget alpha\n")
    P("| alpha | " + " | ".join(PAPER_NAME[v] for v in PAPER_VARS) + " |")
    P("|---|" + "---|" * len(PAPER_VARS))
    for a in PERFOLD_ALPHAS:
        cells = []
        for v in PAPER_VARS:
            m = macro(v, a)
            cells.append(f"{m['zs_acc']:.4f} ({100 * m['fpr']:.2f}%)")
        P(f"| {100 * a:.1f}% | " + " | ".join(cells) + " |")
    P("| pAUC[0,3%] | " + " | ".join(f"{metrics.pauc(S, v):.4f}" for v in PAPER_VARS) + " |")
    P("\nCell = ZS-Acc (realized FPR on val_binary).\n")

    for a in (0.004, 0.015):
        P(f"## T-B. Known-attack detection on val_binary at alpha = {100 * a:.1f}% (macro over 14 rounds)\n")
        P("| Model | Acc | Precision | Recall | F1 | FPR | ZS-Acc |")
        P("|---|---|---|---|---|---|---|")
        for v in PAPER_VARS:
            m = macro(v, a)
            P(f"| {PAPER_NAME[v]} | {m['acc']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
              f"{m['fpr']:.4f} | {m['zs_acc']:.4f} |")
        P("")

    P("## T-C. Reference: F1-max threshold tuned on train (supervised models only)\n")
    P("| Model | tau | Acc | Precision | Recall | F1 | FPR | ZS-Acc |")
    P("|---|---|---|---|---|---|---|---|")
    art = paths.L1_ART
    for v in ["xgboost_only", "clad", "ae_xgboost_mono", "svdd_xgboost_mono"]:
        if v == "clad":
            t = float(np.mean([json.loads((art / "clad" / f / "checkpoint" / "meta.json").read_text())["threshold"]
                               for f in folds]))
        else:
            t = json.loads((art / v / "metrics.json").read_text())["threshold_mean_used_for_eval"]
        rows = [metrics.fold_metrics(S[f"{v}|{f}|vb"], S[f"{v}|{f}|va"], S[f"{v}|{f}|zs"], t) for f in folds]
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        P(f"| {PAPER_NAME[v]} | {t:.4f} | {m['acc']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
          f"{m['f1']:.4f} | {m['fpr']:.4f} | {m['zs_acc']:.4f} |")
    P("")

    for a in (0.004, 0.015):
        P(f"## T-D. Per-type zero-shot accuracy at alpha = {100 * a:.1f}%\n")
        P("| Withheld type | n test | " + " | ".join(PAPER_NAME[v] for v in PAPER_VARS) + " |")
        P("|---|---|" + "---|" * len(PAPER_VARS))
        for f in folds:
            n = len(S[f"xgboost_only|{f}|zs"])
            P(f"| {f} | {n} | " + " | ".join(
                f"{(S[f'{v}|{f}|zs'] >= metrics.fold_quantile(S[f'{v}|{f}|tb'], a)).mean():.4f}"
                for v in PAPER_VARS) + " |")
        P("")

    txt = "\n".join(out)
    (paths.ensure(paths.OUT_L1) / "layer1_tables.md").write_text(txt, encoding="utf-8")
    return txt


# ------------------------------------------------------------------ tai hien danh gia cua nguoi train
def _metrics_file(key):
    """metrics.json cua nguoi train, nam canh checkpoint (deep_svdd_paper/ khong co: Deep SVDD khong kiem duoc)."""
    root = paths.BASE_ART / scorers.MODELS[key].dir
    for cand in [root / "metrics.json"] + sorted(root.glob("*metrics*.json")):
        if cand.exists():
            return cand
    return None


def own_threshold(key, m, fold):
    """(nguong, strict) ma nguoi train da dung; strict=True nghia la alert khi diem > nguong."""
    if "threshold_mean_used_for_eval" in m:                      # ho XGBoost, LightGBM, KNN
        return m["threshold_mean_used_for_eval"], False
    if key == "clad":                                            # F1-max tren val, tung fold
        meta = paths.BASE_ART / "clad" / fold / "checkpoint" / "meta.json"
        return json.loads(meta.read_text(encoding="utf-8"))["threshold"], False
    if "optiforest_threshold" in m:                              # Q_0.995 tren benign train, diem >= nguong
        return m["optiforest_threshold"], False
    for k in ("threshold_per_label_from_benign_train", "threshold_per_label"):
        if k in m:
            return m[k][fold], True
    for k in ("ae_threshold", "vae_threshold", "threshold_benign_train", "svdd_threshold"):
        if k in m:
            return m[k], True
    raise KeyError(f"{key}: khong co truong nguong trong metrics.json")


def verify_reproduction():
    """Ap nguong rieng cua tung nguoi train (metrics.json) len diem da cham lai, so voi tom tat ho bao cao.
    Khop 4 chu so nghia la duong cham diem cua scorers.py tai hien dung danh gia goc, nen cac so o nguong chung
    cua tables.py la cua chinh cac model do."""
    lines = ["| Model | Trainer's rule | ZS-Acc reported / rescored | KA-F1 reported / rescored | "
             "KA-FPR reported / rescored | Match |", "|---|---|---|---|---|---|"]
    for key, spec in scorers.MODELS.items():
        S, mf = scores.load(key, scores.BASE), _metrics_file(key)
        if S is None or mf is None:
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        rows = []
        for f in metrics.folds_of(S):
            t, strict = own_threshold(key, m, f)
            hit = (lambda a: a > t) if strict else (lambda a: a >= t)
            vb, va, zs = (S[f"{key}|{f}|{p}"] for p in ("vb", "va", "zs"))
            tp, fp = int(hit(va).sum()), int(hit(vb).sum())
            pre, rec = (tp / (tp + fp) if tp + fp else 0.0), tp / len(va)
            rows.append({"zs": float(hit(zs).mean()), "f1": 2 * pre * rec / (pre + rec) if pre + rec else 0.0,
                         "fpr": fp / len(vb)})
        got = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        rep = {"zs": m["summary"]["zero_shot_macro_avg"]["accuracy"],
               "f1": m["summary"]["known_attack_macro_avg"]["f1"],
               "fpr": m["summary"]["known_attack_macro_avg"]["fpr"]}
        ok = all(abs(got[k] - rep[k]) < 5e-4 for k in rep)
        strict0 = own_threshold(key, m, metrics.folds_of(S)[0])[1]
        lines.append(f"| {spec.label} | {'>' if strict0 else '>='} {mf.name} threshold | "
                     f"{rep['zs']:.4f} / {got['zs']:.4f} | {rep['f1']:.4f} / {got['f1']:.4f} | "
                     f"{rep['fpr']:.4f} / {got['fpr']:.4f} | {'yes' if ok else 'NO'} |")
    text = "## Reproduction of each trainer's own evaluation\n\n" + "\n".join(lines) + "\n"
    (paths.ensure(paths.OUT_BASE) / "verify.md").write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------------ hai run train cua cung model
RUN_PAIRS = [   # ten, khoa trong run cua paper, khoa trong run cua baseline study
    ("XGBoost-only", "xgboost_only", "xgboost_only"),
    ("AE-only", "ae_only", "ae_only"),
    ("Mixture Deep SVDD", "svdd_only", "svdd_only"),
    ("CLAD", "clad", "clad"),
    ("AE-XGBoost", "ae_xgboost_mono", "ae_xgboost"),
    ("SVDD-XGBoost", "svdd_xgboost_mono", "svdd_xgboost"),
]


def checkpoint_runs():
    """Cac model co trong ca hai run train, cham cung quy tac cua tables.py. Cot "same" = diem bang nhau o moi
    fold (cung mot checkpoint, khong phai train lai). Day la co so cua lua chon checkpoint trong tables.ROWS."""
    out = []
    for name, kp, kb in RUN_PAIRS:
        Sp, Sb = scores.get(kp, scores.PAPER), scores.get(kb, scores.BASE)
        same = all(np.allclose(Sp[f"{kp}|{f}|{p}"], Sb[f"{kb}|{f}|{p}"], rtol=0, atol=1e-6)
                   for f in metrics.folds_of(Sp) for p in ("tb", "vb", "va", "zs"))
        for run, key, S in (("paper", kp, Sp), ("baseline-study", kb, Sb)):
            t = metrics.evaluate(S, key, tables.ALPHA_TABLE, tables.RULE)[0]
            out.append({"model": name, "run": run, "key": key, "f1": t["f1"], "fpr": t["fpr"],
                        "zs_acc": t["zs_acc"], "pauc": metrics.pauc(S, key), "same": same})
    return pd.DataFrame(out).set_index(["model", "run"])
