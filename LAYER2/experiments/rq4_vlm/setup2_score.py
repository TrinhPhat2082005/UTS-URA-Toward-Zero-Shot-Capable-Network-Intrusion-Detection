"""
Cham Setup 2 (paper Table tab:res-vlm-zeroday). Moi alert trong Setup 2 thuoc loai bi giu
lai, khong co trong danh sach lua chon -> dap an dung la UNKNOWN.

  co co     : ty le UNKNOWN (giu ket luan "chua biet"), ty le gan nham loai da biet
  khong co  : ty le UNKNOWN = phan alert la VLM cuu duoc sau khi retriever bo sot;
              gan nham co trung top-1 cua retriever khong (lap lai loi retriever)
  tactic    : tactic VLM neu co thuoc tactic that cua loai bi giu lai; baseline = tactic
              cua top-1 retriever; tach fold DoS (Impact, baseline dung san) va con lai
  vi pham   : chon UNKNOWN ma van neu technique ID
  schema    : du 6 muc; grounding: top indicators nam trong 5 attribution lon nhat
  end-to-end: ZS-Acc x ti le co x ti le UNKNOWN (nhom co), va ban tinh ca nhom khong co;
              macro tren 12 loai, fold khong du alert tinh = 0 (giong tab:res-novelty)

CHAY:  python LAYER2/experiments/rq4_vlm/setup2_score.py <results.jsonl> <bundle_dir> <out_dir>
Output: summary.md, summary.json, per_fold.csv, per_alert.csv
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

UNKNOWN = "UNKNOWN"
DOS_TACTIC = "Impact"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _norm(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def load(results_path):
    rows = [json.loads(l) for l in open(results_path, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    df["ok"] = df["run_error"].isna() if "run_error" in df else True
    df["sel_unknown"] = df["selected"] == UNKNOWN
    df["sel_known"] = df["selected"].notna() & ~df["sel_unknown"]
    df["sel_invalid"] = df["selected"].isna()
    df["repeat_top1"] = df["sel_known"] & (df["selected"] == df["retriever_top1"])
    df["tactic_ok"] = [t in tt for t, tt in zip(df["tactic"], df["true_tactics"])]
    df["tactic_base_ok"] = [bool(set(b) & set(tt)) for b, tt in zip(df["retriever_top1_tactics"], df["true_tactics"])]
    df["tech_when_unknown"] = df["sel_unknown"] & df["technique_ids"].map(lambda x: len(x or []) > 0)
    df["schema_ok"] = df["n_sections"] == 6
    df["dos_fold"] = df["true_tactics"].map(lambda t: list(t) == [DOS_TACTIC])

    def grounded(r):
        ind = [x.strip() for x in str(r.get("top_indicators") or "").split(",") if x.strip()]
        top = {_norm(f["name"]) for f in r["shap_top_features"]}
        return float(np.mean([_norm(x) in top for x in ind])) if ind else np.nan
    df["grounding_top5"] = df.apply(grounded, axis=1)
    return df


def rate(s):
    s = s.dropna().astype(float)
    return (int(s.sum()), int(len(s)), float(s.mean()) if len(s) else float("nan"))


def fmt(k, n, p):
    if n == 0:
        return "—"
    lo, hi = wilson(k, n)
    return f"{p:.1%} ({k}/{n}) [{lo:.1%}, {hi:.1%}]"


def score(results_path, bundle_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load(results_path)
    stats = json.loads((Path(bundle_dir) / "fold_stats.json").read_text(encoding="utf-8"))
    d = df[df["ok"]]
    groups = {"co co": d[d["flagged"]], "khong co": d[~d["flagged"]], "tat ca": d}

    lines = [f"# Setup 2 — ket qua ({len(d)} bao cao chay duoc / {len(df)} ban ghi)\n",
             "| Chi so | Co co | Khong co | Tat ca |", "|---|---|---|---|"]
    summary = {}

    def row(name, col, base=None):
        cells, js = [], {}
        for g, x in groups.items():
            src = x if base is None else x[base(x)]
            k, n, p = rate(src[col])
            cells.append(fmt(k, n, p))
            js[g] = {"k": k, "n": n, "rate": p}
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
        summary[name] = js

    row("Chon UNKNOWN (dung)", "sel_unknown")
    row("Gan nham loai da biet", "sel_known")
    row("  trong do trung top-1 retriever", "repeat_top1", lambda x: x["sel_known"])
    row("Lua chon khong hop le", "sel_invalid")
    row("Chon UNKNOWN nhung van neu technique", "tech_when_unknown", lambda x: x["sel_unknown"])
    row("Tactic dung (VLM)", "tactic_ok")
    row("Tactic dung (baseline top-1 retriever)", "tactic_base_ok")
    row("Tactic dung (VLM), fold khong phai DoS", "tactic_ok", lambda x: ~x["dos_fold"])
    row("Tactic dung (baseline), fold khong phai DoS", "tactic_base_ok", lambda x: ~x["dos_fold"])
    row("Du 6 muc", "schema_ok")
    gr = d["grounding_top5"].dropna()
    lines.append(f"| Grounding (indicator trong top-5 SHAP), trung binh | | | {gr.mean():.1%} (n={len(gr)}) |")

    per = []
    for fold, st in sorted(stats.items()):
        x = d[d["fold"] == fold]
        xf, xu = x[x["flagged"]], x[~x["flagged"]]
        uf = float(xf["sel_unknown"].mean()) if len(xf) else float("nan")
        uu = float(xu["sel_unknown"].mean()) if len(xu) else float("nan")
        fr = st.get("flag_rate", float("nan"))
        if st.get("excluded") or not len(x):
            e2e_flag = e2e_all = 0.0
        else:
            e2e_flag = st["zs_acc"] * fr * (uf if len(xf) else 0.0)
            e2e_all = st["zs_acc"] * (fr * (uf if len(xf) else 0.0) + (1 - fr) * (uu if len(xu) else 0.0))
        per.append({"fold": fold, "twin": st["twin"], "excluded": st.get("excluded"),
                    "n_test": st["n_test"], "zs_acc": st["zs_acc"], "flag_rate": fr,
                    "n_flagged": len(xf), "n_unflagged": len(xu),
                    "unknown_flagged": uf, "unknown_unflagged": uu,
                    "tactic_vlm": float(x["tactic_ok"].mean()) if len(x) else float("nan"),
                    "tactic_baseline": float(x["tactic_base_ok"].mean()) if len(x) else float("nan"),
                    "e2e_flag_x_unknown": e2e_flag, "e2e_incl_unflagged": e2e_all})
    per = pd.DataFrame(per)
    m12 = per[~per["twin"]]
    summary["end_to_end_macro12"] = {
        "gate_x_flag_only": float((m12["zs_acc"] * m12["flag_rate"].fillna(0)).where(m12["excluded"].isna(), 0).mean()),
        "flag_x_unknown": float(m12["e2e_flag_x_unknown"].mean()),
        "incl_unflagged": float(m12["e2e_incl_unflagged"].mean())}
    e = summary["end_to_end_macro12"]
    lines += ["", "## End-to-end (macro 12 loai, fold khong du alert = 0)", "",
              f"- Layer 1 bat duoc x co zero-day (chua co VLM): {e['gate_x_flag_only']:.4f}",
              f"- ... x VLM chon UNKNOWN (nhom co co): {e['flag_x_unknown']:.4f}",
              f"- tinh ca nhom khong co (VLM cuu duoc): {e['incl_unflagged']:.4f}",
              "", "## Theo fold", "", per.to_string(index=False, float_format=lambda v: f"{v:.3f}")]
    if "t_vlm_s" in d and d["t_vlm_s"].notna().any():
        t = d["t_vlm_s"].dropna()
        summary["latency_vlm_s"] = {"mean": float(t.mean()), "p95": float(np.percentile(t, 95)), "n": int(len(t))}
        lines += ["", f"Latency VLM: trung binh {t.mean():.1f} s, p95 {np.percentile(t, 95):.1f} s (n = {len(t)})"]

    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    per.to_csv(out / "per_fold.csv", index=False)
    keep = ["alert_id", "fold", "flagged", "novelty_score", "tau_zd", "retriever_top1", "selected",
            "selected_raw", "tactic", "tactic_raw", "true_tactics", "technique_ids", "sel_unknown",
            "repeat_top1", "tactic_ok", "tactic_base_ok", "tech_when_unknown", "schema_ok",
            "grounding_top5", "parse_error", "run_error", "t_vlm_s"]
    df[[c for c in keep if c in df]].to_csv(out / "per_alert.csv", index=False)
    print("\n".join(lines))
    return summary, per


if __name__ == "__main__":
    score(*sys.argv[1:4])
