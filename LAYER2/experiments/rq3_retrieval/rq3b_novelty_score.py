"""Cham R3 tren pool DAY DU (r3_full/). Moi round: loc alert theo nguong alpha, novelty = kNN-10 L2.
In va ghi R3_full_tables.md: macro AUROC / Detect@5% cho 3 khoa, Wilcoxon theo round, bang tung round,
ti le trung lap, tactic consistency, end-to-end zero-day."""
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
from shap_knn_retriever import LABEL_TO_PROFILE_ID  # noqa: E402
PROF = paths.PROFILES
TAC = {l: json.loads((PROF / f"{p}.json").read_text(encoding="utf-8"))["mitre_tactic"]
       for l, p in LABEL_TO_PROFILE_ID.items() if (PROF / f"{p}.json").exists()}
S = np.load(paths.OUT_L1 / "sweep_all_scores.npz")  # tu rq1_rq2_layer1/03
VARS = ["xgboost_only", "ae_xgboost_mono", "svdd_xgboost_mono"]
NAME = {"xgboost_only": "XGBoost-only", "ae_xgboost_mono": "AE-XGBoost", "svdd_xgboost_mono": "SVDD-XGBoost"}
TW = {"DoS_attacks_SlowHTTPTest", "FTP_BruteForce"}
KEYS = [("full", "SHAP day du"), ("b78", "SHAP 78"), ("raw", "Feature tho")]
MIN_N = 20
LABELS = sorted({p.stem.split("__")[1] for p in F.glob("*.npz")})
out = []
P = lambda s="": (print(s), out.append(s))
au = lambda u, k: roc_auc_score(np.r_[np.ones(len(u)), np.zeros(len(k))], np.r_[u, k])
d5 = lambda u, k: float((u > np.quantile(k, 0.95)).mean())

for bi, a in ((0, 0.004), (1, 0.015)):
    P(f"\n## alpha = {a:.1%}\n")
    for v in VARS:
        rows = []
        for l in LABELS:
            D = np.load(F / f"{v}__{l}.npz", allow_pickle=True)
            t = D["thr_values"][bi]
            km, um = D["p_known"] >= t, D["p_unk"] >= t
            zs = float((S[f"{v}|{l}|zs"] >= np.quantile(S[f"{v}|{l}|tb"], 1 - a, method="higher")).mean())
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
        e2e = np.mean([r["zs"] * r.get("full_d5", 0.0) for r in rows])
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

(HERE / "R3_full_tables.md").write_text("\n".join(out), encoding="utf-8")
print("\nsaved R3_full_tables.md")
