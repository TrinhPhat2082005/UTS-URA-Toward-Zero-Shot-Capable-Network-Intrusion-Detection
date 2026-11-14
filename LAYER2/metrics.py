"""
Tong hop mot lan chay VLM (thu muc RUN_DIR cua pipeline.py): ket qua tung sample + latency tung chang.

    res, lat = load_run(config.RUN_DIR)
    summary(res)        # ty le dung lop / technique, trich dan hop le, schema, theo nhanh co
    per_label(res)      # theo nhan that
    latency_table(lat)  # mean / median / p95 tung chang (ms)

Danh gia ngu nghia (faithfulness, answer relevancy...) chay bang RAGAS tren ragas.jsonl (user_input,
retrieved_contexts, response); file nay chi tinh cac chi so co the kiem bang code:
    class_correct       co tat: Predicted class = nhan that; co bat: Predicted class = UNKNOWN
    technique_ok        co tat: technique o muc 4 thuoc technique cua nhan (Borisenko); co bat: khong co T-ID
    citation_precision  ty le trich dan [I/E/C/K:..] co that trong prompt (groundedness o muc id)
    mitigations_valid   muc 5 chi chon trong ung vien G cua alert
"""
from __future__ import annotations

import json
import os

import pandas as pd

from pipeline import STAGES


def load_run(run_dir):
    rows = []
    with open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                rows.append(json.loads(ln))
    res = pd.DataFrame(rows).drop_duplicates("sample_id", keep="last")
    lat_p = os.path.join(run_dir, "latency.csv")
    lat = pd.read_csv(lat_p).drop_duplicates("sample_id", keep="last") if os.path.exists(lat_p) else None
    print(f"{len(res)} sample ({res['run_error'].notna().sum()} loi) tu {run_dir}")
    return res, lat


def _rate(s):
    s = s.dropna()
    return f"{s.mean():.3f} ({int(s.sum())}/{len(s)})" if len(s) else "-"


def summary(res):
    ok = res[res["run_error"].isna() & res["alert"].fillna(False).astype(bool)]
    flag = ok["flagged"].fillna(False).astype(bool)
    rows = []
    for name, g in [("all", ok), ("flag off", ok[~flag]), ("flag on", ok[flag])]:
        r = {"subset": name, "n": len(g),
             "lookup label = true": _rate((g["pred_label"] == g["label_true"]).astype(float))}
        for c in ("class_correct", "technique_ok", "mitigations_valid"):
            r[c] = _rate(g[c].astype(float)) if c in g else "-"
        r["citation_precision (mean)"] = round(g["citation_precision"].mean(), 3) if "citation_precision" in g else "-"
        r["citation_coverage (mean)"] = round(g["citation_coverage"].mean(), 3) if "citation_coverage" in g else "-"
        r["unknown citations (sum)"] = int(g["n_unknown_citations"].sum()) if "n_unknown_citations" in g else "-"
        r["6 sections"] = _rate((g["n_sections"] == 6).astype(float)) if "n_sections" in g else "-"
        rows.append(r)
    return pd.DataFrame(rows).set_index("subset")


def per_label(res):
    ok = res[res["run_error"].isna()].copy()
    ok["lookup_ok"] = (ok["pred_label"] == ok["label_true"]).astype(float)
    cols = {"n": ("sample_id", "count"), "alert": ("alert", "mean"), "flagged": ("flagged", "mean"),
            "lookup_ok": ("lookup_ok", "mean")}
    for c in ("class_correct", "technique_ok", "citation_precision", "citation_coverage", "mitigations_valid"):
        if c in ok:
            ok[c] = ok[c].astype(float)
            cols[c] = (c, "mean")
    for c in ("alert", "flagged"):
        ok[c] = ok[c].fillna(False).astype(float)
    return ok.groupby("label_true").agg(**cols).round(3)


def latency_table(lat):
    """ms; chi tinh sample co alert (chang nao khong chay thi bo qua)."""
    if lat is None:
        return None
    lat = lat[lat["alert"].astype(str) == "True"]
    rows = []
    for s in [*STAGES, "total"]:
        v = pd.to_numeric(lat[s], errors="coerce").dropna() * 1000
        if len(v):
            rows.append({"stage": s, "n": len(v), "mean_ms": v.mean(), "median_ms": v.median(),
                         "p95_ms": v.quantile(0.95), "max_ms": v.max()})
    return pd.DataFrame(rows).set_index("stage").round(2)
