"""Cham R3 tren pool DAY DU (r3_full/). Moi round: loc alert theo nguong alpha, novelty = kNN-10 L2.
In va ghi R3_full_tables.md: macro AUROC / Detect@5% cho 3 khoa, Wilcoxon theo round, bang tung round,
ti le trung lap, tactic consistency, end-to-end zero-day.

Mac dinh: alpha = 0.4% va 1.5%, nguong rieng tung fold (nhu paper).
Ngan sach / quy tac khac (alert da luu o nguong thap nhat cua build, nen chi can alpha <= 1.5%):
    python LAYER2/experiments/rq3_retrieval/rq3b_novelty_score.py --alpha 0.005 --rule meanfold
  meanfold = trung binh 14 nguong Q_(1-alpha) cua benign train, mot nguong cho moi fold
  (quy tac cua bang Layer 1, LAYER1/04_experiments/layer1_eval/tables.py). Output: R3_full_tables_alpha0.5_meanfold.md"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402  (experiments/paths.py)
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

HERE = paths.ensure(paths.OUT_RQ3B)
F = HERE / "r3_full"
paths.use_layer2_code()
# Tactic cua moi nhan theo Borisenko et al. 2022 Table II (knowledge_base/store/index.json); nhan co hai entry
# (Infiltration, Bot) co hai tactic -> so sanh theo tap tactic
_KB_INDEX = json.loads((paths.LAYER2 / "knowledge_base" / "store" / "index.json").read_text(encoding="utf-8"))
TAC = {l: tuple(sorted({e["tactic"] for e in v["entries"]})) for l, v in _KB_INDEX["labels"].items()}
S = dict(np.load(paths.L1_SCORES / "paper.npz"))  # diem 6 model paper, tu LAYER1/04_experiments
VARS = ["xgboost_only", "ae_xgboost_mono", "svdd_xgboost_mono"]
NAME = {"xgboost_only": "XGBoost-only", "ae_xgboost_mono": "AE-XGBoost", "svdd_xgboost_mono": "SVDD-XGBoost",
        "svdd_xgboost_bs": "SVDD-XGBoost (baseline study)"}
# SVDD-XGBoost cua baseline study (rq3b_novelty_build.py --variants svdd_xgboost_bs), diem tu LAYER1/output/scores
if any((paths.OUT_RQ3B / "r3_full").glob("svdd_xgboost_bs__*.npz")):
    VARS.append("svdd_xgboost_bs")
    _bs = np.load(paths.L1_SCORES / "baseline_svdd_xgboost.npz")
    S.update({k.replace("svdd_xgboost|", "svdd_xgboost_bs|"): _bs[k] for k in _bs.files})
TW = {"DoS_attacks_SlowHTTPTest", "FTP_BruteForce"}
KEYS = [("full", "SHAP day du"), ("b78", "SHAP 78"), ("raw", "Feature tho")]
MIN_N = 20
LABELS = sorted({p.stem.split("__")[1] for p in F.glob("*.npz")})
out = []
P = lambda s="": (print(s), out.append(s))
au = lambda u, k: roc_auc_score(np.r_[np.ones(len(u)), np.zeros(len(k))], np.r_[u, k])
d5 = lambda u, k: float((u > np.quantile(k, 0.95)).mean())

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, default=None)
ap.add_argument("--rule", choices=["perfold", "meanfold"], default="perfold")
args = ap.parse_args()
q = lambda v, l, a: np.quantile(S[f"{v}|{l}|tb"], 1 - a, method="higher")


def tau(v, l, a, bi):
    """Nguong Layer 1 cua fold l. Mac dinh: nguong da luu luc build (bi)."""
    if args.alpha is None:
        return np.load(F / f"{v}__{l}.npz")["thr_values"][bi]
    return q(v, l, a) if args.rule == "perfold" else float(np.mean([q(v, x, a) for x in LABELS]))


RUNS = ((0, 0.004), (1, 0.015)) if args.alpha is None else ((None, args.alpha),)
for bi, a in RUNS:
    P(f"\n## alpha = {a:.1%}" + ("" if args.alpha is None else f", rule = {args.rule}") + "\n")
    for v in VARS:
        rows = []
        for l in LABELS:
            D = np.load(F / f"{v}__{l}.npz", allow_pickle=True)
            t = tau(v, l, a, bi)
            assert t >= D["thr_values"].min() - 1e-9, f"{v}/{l}: nguong {t} thap hon nguong luc build"
            km, um = D["p_known"] >= t, D["p_unk"] >= t
            zs = float((S[f"{v}|{l}|zs"] >= t).mean())
            r = {"lab": l, "n_unk": int(um.sum()), "n_known": int(km.sum()), "zs": zs, "twin": l in TW,
                 "n_pool": int(D["n_pool"])}
            if um.sum() >= MIN_N and km.sum() >= MIN_N:
                for k, _ in KEYS:
                    u, kk = D[f"s_{k}_unk"][um], D[f"s_{k}_known"][km]
                    r[k] = au(u, kk); r[k + "_d5"] = d5(u, kk)
                r["dup_unk"] = float((D["dmin_unk"][um] < 1e-4).mean())
                r["dup_known"] = float((D["dmin_known"][km] < 1e-4).mean())
                r["tac_full"] = float(np.mean([TAC.get(n) == TAC.get(l) for n in D["nb_full_unk"][um]]))
                r["tac_raw"] = float(np.mean([TAC.get(n) == TAC.get(l) for n in D["nb_raw_unk"][um]]))
            rows.append(r)
        ev = [r for r in rows if "full" in r]
        ex = [r for r in ev if not r["twin"]]
        P(f"### {NAME[v]} — {len(ev)} round hop le (pool = toan bo attack train cua round, {min(r['n_pool'] for r in rows):,}–{max(r['n_pool'] for r in rows):,} flow)\n")
        P("| Khoa | AUROC macro (tat ca) | AUROC macro (bo song sinh) | Detect@5% macro |")
        P("|---|---|---|---|")
        for k, nm in KEYS:
            P(f"| {nm} | {np.mean([r[k] for r in ev]):.3f} | {np.mean([r[k] for r in ex]):.3f} | {np.mean([r[k + '_d5'] for r in ev]):.3f} |")
        P("")
        for (k1, n1), (k2, n2) in [(KEYS[0], KEYS[2]), (KEYS[1], KEYS[2]), (KEYS[0], KEYS[1])]:
            x, y = np.array([r[k1] for r in ev]), np.array([r[k2] for r in ev])
            p = wilcoxon(x, y, zero_method="zsplit").pvalue
            P(f"- {n1} vs {n2}: chenh TB {np.mean(x - y):+.3f}, hon o {int((x > y).sum())}/{len(x)} round, Wilcoxon p = {p:.3f}")
        # Quy uoc cua paper (tab:res-novelty): chi cac round hop le KHONG phai song sinh
        P(f"\n**Paper (tab:res-novelty) — {len(ex)} round, bo song sinh:**\n")
        P("| Khoa | AUROC | Detect@5% |")
        P("|---|---|---|")
        for k, nm in KEYS:
            P(f"| {nm} | {np.mean([r[k] for r in ex]):.3f} | {np.mean([r[k + '_d5'] for r in ex]):.3f} |")
        for m, sfx in (("AUROC", ""), ("Detect@5%", "_d5")):
            x = np.array([r["full" + sfx] for r in ex])
            y = np.array([r["raw" + sfx] for r in ex])
            P(f"- {m} SHAP day du vs Feature tho: Wilcoxon p = {wilcoxon(x, y, zero_method='zsplit').pvalue:.3f}")
        P("")
        e2e =np.mean([r["zs"] * r.get("full_d5", 0.0) for r in rows])
        e2e_ex = np.mean([r["zs"] * r.get("full_d5", 0.0) for r in rows if not r["twin"]])
        P(f"- End-to-end zero-day (ZS-Acc x Detect@5%, SHAP day du, 14 round, round khong hop le = 0): {e2e:.4f} (bo song sinh: {e2e_ex:.4f})\n")
        P("| Round | n unk | n known | ZS-Acc | Trung lap (unk) | AUROC SHAP day du | AUROC SHAP 78 | AUROC tho | Det@5% SHAP | Tactic (SHAP) | Tactic (tho) |")
        P("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if "full" not in r:
                P(f"| {r['lab']} | {r['n_unk']} | {r['n_known']} | {r['zs']:.3f} | — | — | — | — | — | — | — |")
                continue
            P(f"| {r['lab']}{' ‡' if r['twin'] else ''} | {r['n_unk']:,} | {r['n_known']:,} | {r['zs']:.3f} | {r['dup_unk']:.3f} | "
              f"{r['full']:.3f} | {r['b78']:.3f} | {r['raw']:.3f} | {r['full_d5']:.3f} | {r['tac_full']:.3f} | {r['tac_raw']:.3f} |")
        P("")

name = "R3_full_tables" + ("" if args.alpha is None else f"_alpha{100 * args.alpha:g}_{args.rule}") + ".md"
(HERE / name).write_text("\n".join(out), encoding="utf-8")
print(f"\nsaved {name}")
