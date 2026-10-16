"""Cham Setup 1 (Protocol B). Nhan dung luon nam trong ba ung vien, nen cau hoi la
VLM co chon dung hon ung vien gan nhat khong.

  moc so sanh : do chinh xac top-1 cua retriever tren dung tap alert nay (bundle_meta)
  VLM         : ty le chon dung, kem so ca sua duoc va pha hong cua retriever
  bao cao     : du 6 muc; grounding: top indicators nam trong 5 attribution lon nhat

CHAY:  python LAYER2/experiments/rq4_vlm/setup1_score.py <results.jsonl> <bundle_dir> <out_dir>
Output: summary.md, summary.json, per_label.csv, per_alert.csv
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SECTIONS = ["EXECUTIVE SUMMARY", "ATTACK ASSESSMENT", "FORENSIC REASONING",
            "MITRE MAPPING", "MITIGATIONS", "ANALYST NOTES"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _norm(s):
    return str(s).strip().lower().replace("-", "_").replace(" ", "_") if s else s


def load(results_path):
    rows = [json.loads(l) for l in Path(results_path).open(encoding="utf-8") if l.strip()]
    d = pd.DataFrame(rows)
    n = len(d)
    sel = d["selected"] if "selected" in d else pd.Series([None] * n)
    d["sel_ok"] = [_norm(s) == _norm(t) for s, t in zip(sel, d["true_label"])]
    d["ret_ok"] = [_norm(s) == _norm(t) for s, t in zip(d["retriever_top1"], d["true_label"])]

    raw = (d["raw_output"] if "raw_output" in d else pd.Series([""] * n)).fillna("")
    d["schema_ok"] = [sum(s in str(r).upper() for s in SECTIONS) == len(SECTIONS) for r in raw]
    # Grounding: ba chi bao hang dau phai co that trong 5 attribution lon nhat cua alert.
    top5 = [{f["name"] for f in (r or [])[:5]} for r in
            (d["shap_top_features"] if "shap_top_features" in d else [[]] * n)]
    hits = [re.search(r"top\s*indicators\s*[:：]\s*(.+)", str(r), re.IGNORECASE) for r in raw]
    d["grounding_top5"] = [
        all(any(_norm(x) == _norm(f) for f in t) for x in [p.strip() for p in m.group(1).split(",")][:3])
        if m and t else False for m, t in zip(hits, top5)]
    return d


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{(k / n if n else 0):.4f} ({k}/{n}, 95% CI {lo:.3f}-{hi:.3f})"


def score(results_path, bundle_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = load(results_path)
    meta = json.loads((Path(bundle_dir) / "bundle_meta.json").read_text(encoding="utf-8"))
    n = len(d)
    n_sel, n_ret = int(d["sel_ok"].sum()), int(d["ret_ok"].sum())
    broke = int((d["ret_ok"] & ~d["sel_ok"]).sum())
    fixed = int((~d["ret_ok"] & d["sel_ok"]).sum())
    opts = d["options"] if "options" in d else pd.Series([[]] * n)
    sel = d["selected"] if "selected" in d else pd.Series([None] * n)
    in_opts = int(sum(any(_norm(s) == _norm(o) for o in (o_ or [])) for s, o_ in zip(sel, opts)))
    perr = int((d["parse_error"].notna().sum()) if "parse_error" in d else 0)
    rerr = int((d["run_error"].notna().sum()) if "run_error" in d else 0)

    summary = {
        "variant": meta.get("variant"), "n_alerts": n,
        "retriever_top1": {"accuracy": n_ret / n, "n_correct": n_ret},
        "vlm": {"accuracy": n_sel / n, "n_correct": n_sel},
        "delta_vs_retriever": (n_sel - n_ret) / n,
        "broke": broke, "fixed": fixed, "net": fixed - broke,
        "selection_in_options": in_opts / n,
        "report": {"schema_ok": float(d["schema_ok"].mean()),
                   "grounding_top5": float(d["grounding_top5"].mean())},
        "parse_errors": perr, "run_errors": rerr,
    }
    per = (d.groupby("true_label")
             .agg(n=("sel_ok", "size"), vlm=("sel_ok", "mean"), retriever=("ret_ok", "mean"))
             .reset_index().sort_values("true_label"))

    lines = [f"# Setup 1 - {meta.get('variant')} ({n} alert, Protocol B)", "",
             "Nhan dung luon nam trong ba ung vien (Hit@3 = "
             f"{meta.get('retriever_hit_at_3', 1.0):.1%}), nen moc can vuot la top-1 cua retriever.",
             "", "## Nhan dien", "",
             f"- retriever top-1 : {fmt(n_ret, n)}",
             f"- VLM             : {fmt(n_sel, n)}",
             f"- chenh lech      : {summary['delta_vs_retriever']:+.4f}",
             f"- VLM pha (retriever dung -> VLM sai): {broke}",
             f"- VLM sua (retriever sai -> VLM dung): {fixed}",
             f"- net             : {fixed - broke:+d}", "",
             "## Tuan thu va bao cao", "",
             f"- chon trong danh sach: {fmt(in_opts, n)}",
             f"- du 6 muc            : {fmt(int(d['schema_ok'].sum()), n)}",
             f"- grounding top-5     : {fmt(int(d['grounding_top5'].sum()), n)}",
             f"- parse error         : {perr}",
             f"- run error           : {rerr}", "",
             "## Theo loai", "",
             per.to_string(index=False, float_format=lambda v: f"{v:.3f}")]
    if "t_vlm_s" in d and d["t_vlm_s"].notna().any():
        t = d["t_vlm_s"].dropna()
        summary["latency_vlm_s"] = {"mean": float(t.mean()), "p95": float(np.percentile(t, 95)),
                                    "n": int(len(t))}
        lines += ["", f"Latency VLM: trung binh {t.mean():.1f} s, p95 {np.percentile(t, 95):.1f} s "
                      f"(n = {len(t)})"]

    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    per.to_csv(out / "per_label.csv", index=False)
    keep = ["alert_id", "true_label", "retriever_top1", "selected", "selected_raw", "ret_ok",
            "sel_ok", "tactic", "schema_ok", "grounding_top5", "novelty_score", "flagged",
            "parse_error", "run_error", "t_vlm_s"]
    d[[c for c in keep if c in d]].to_csv(out / "per_alert.csv", index=False)
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    score(*sys.argv[1:4])
