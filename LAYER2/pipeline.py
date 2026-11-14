"""
Duong di MOT sample qua toan bo Layer 2 va noi ghi ket qua (dung boi notebook vlm_run.ipynb).

    row tho -> EvidenceBuilder.build   layer1 · shap · lookup · contrast · kb_select · images   (evidence/build.py)
            -> prompts.build_report_prompt                                         chang "prompt"
            -> VLMRunner.generate (Qwen2.5-VL, 5 anh)                              chang "vlm"
            -> prompts.parse_report (+ kiem trich dan)                             chang "parse"

Ket qua cua moi sample (sample_id = <Label_Text>__r<chi so dong trong val_full.csv>):

    <out>/<sample_id>/evidence/   shap_waterfall.png + 4 chart_*.png, evidence.json (top-5, tra cuu, bang C, doan KB)
    <out>/<sample_id>/text/       system_prompt.txt, user_prompt.txt (prompt day du truoc khi gui VLM), contexts.json
    <out>/<sample_id>/output/     report.md (nguyen van VLM), report.json (truong da parse + trich dan)
    <out>/<sample_id>/record.json tom tat + latency tung chang (giay)
    <out>/results.jsonl, ragas.jsonl, latency.csv   moi sample mot dong (ghi ngay, chay lai thi bo qua sample da xong)
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
import traceback

import config
import prompts
from utils.io import append_jsonl, git_commit
from utils.timing import StageTimer

STAGES = ("layer1", "shap", "lookup", "contrast", "kb_select", "images", "prompt", "vlm", "parse")


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            json.dump(obj, f, ensure_ascii=False, indent=1, default=str)


def done_ids(out_root):
    """sample_id da chay xong (co trong results.jsonl, khong loi) - de chay tiep sau khi mat ket noi."""
    p = os.path.join(out_root, "results.jsonl")
    if not os.path.exists(p):
        return set()
    out = set()
    with open(p, encoding="utf-8") as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if not r.get("run_error"):
                out.add(r["sample_id"])
    return out


def write_run_meta(out_root, eb, vlm=None, kb_mode=None, kb_parts=None, extra=None):
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "code_commit": git_commit(config.LAYER2_DIR),
            "variant": config.VARIANT, "tau_alpha": float(eb.l1.threshold), "tau_zd": eb.store.tau_zd,
            "reference_set": os.path.basename(config.REFERENCE_SET_NPZ), "n_reference": len(eb.store),
            "kb_sha256": eb.kb.index.get("kb_sha256"), "kb_mode": kb_mode or config.KB_MODE,
            "kb_parts": list(kb_parts or config.KB_PARTS), "exclude_labels": eb.exclude,
            "vlm_model": config.VLM_MODEL_ID if vlm is not None else None, "vlm_4bit": config.VLM_LOAD_4BIT,
            "vlm_max_new_tokens": config.VLM_MAX_NEW_TOKENS, "vlm_max_pixels": config.VLM_MAX_PIXELS,
            "system_prompt": prompts.SYSTEM_PROMPT, **(extra or {})}
    _write(os.path.join(out_root, "run_meta.json"), meta)
    return meta


def warmup(eb, rows, n=3):
    """Chay chang A (khong VLM, khong ghi) tren n dong de nap cache/JIT truoc khi do latency."""
    tmp = tempfile.mkdtemp(prefix="warmup_")
    try:
        for _, row in rows.head(n).iterrows():
            eb.build(row, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _latency_row(out_root, rec):
    p = os.path.join(out_root, "latency.csv")
    new = not os.path.exists(p)
    with open(p, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["sample_id", "label_true", "alert", "flagged", *STAGES, "total"])
        lat = rec.get("latency", {})
        w.writerow([rec["sample_id"], rec.get("label_true"), rec.get("alert"), rec.get("flagged"),
                    *[f"{lat[s]:.6f}" if s in lat else "" for s in STAGES], f"{rec.get('latency_total', 0):.6f}"])


def run_sample(row, eb, vlm, out_root, kb_mode=None, kb_parts=None):
    """Chay toan bo pipeline cho MOT dong va ghi ket qua. vlm=None: dung truoc buoc VLM (kiem prompt local)."""
    kb_mode = kb_mode or config.KB_MODE
    kb_parts = tuple(kb_parts or config.KB_PARTS)
    sid = str(row["sample_id"])
    d = os.path.join(out_root, sid)
    t = StageTimer()
    rec = {"sample_id": sid, "label_true": row.get("Label_Text"), "row_idx": row.get("row_idx"),
           "kb_mode": kb_mode, "kb_parts": list(kb_parts), "run_error": None}
    try:
        ev = eb.build(row, os.path.join(d, "evidence"), timer=t)
        rec.update({"p": ev["p"], "tau_alpha": ev["tau_alpha"], "alert": ev["alert"]})
        if ev["alert"]:
            look = ev["lookup"]
            rec.update({"pred_label": look["label"], "flagged": look["flagged"], "novelty": look["novelty"],
                        "votes": look["votes"], "techniques": ev["kb"]["techniques"],
                        "kb_pids": [p["pid"] for p in ev["kb"]["passages"]],
                        "top_features": [f["name"] for f in ev["top_features"]]})
            ev_json = {**ev, "images": [os.path.relpath(p, d).replace("\\", "/") for p in ev["images"]]}
            _write(os.path.join(d, "evidence", "evidence.json"), ev_json)
            _write(os.path.join(d, "evidence", "top5.json"), ev["top_features"] + ev["x_facts"])

            with t("prompt"):
                pr = prompts.build_report_prompt(ev, kb_mode=kb_mode, kb_parts=kb_parts)
            _write(os.path.join(d, "text", "system_prompt.txt"), pr["system"])
            _write(os.path.join(d, "text", "user_prompt.txt"), pr["user"])
            _write(os.path.join(d, "text", "contexts.json"),
                   {"contexts": pr["contexts"], "allowed_ids": pr["allowed_ids"]})
            rec["prompt_chars"] = len(pr["system"]) + len(pr["user"])

            if vlm is not None:
                with t("vlm"):
                    raw, conf, t_gen = vlm.generate(pr["system"], pr["user"], ev["images"])
                with t("parse"):
                    parsed = prompts.parse_report(raw, pr["allowed_ids"])
                cites, bad = parsed["citations"], set(parsed["unknown_citations"])
                # technique hop le: technique Borisenko cua nhan + technique cha cua sub-technique (Table II ghi ca hai)
                tech_ok = set(ev["kb"]["techniques"]) | {t.split(".")[0] for t in ev["kb"]["techniques"]}
                score = {
                    "n_citations": len(cites),
                    "n_unknown_citations": len(bad),
                    "citation_precision": (sum(c not in bad for c in cites) / len(cites)) if cites else None,
                    "citation_coverage": (parsed["n_cited_units"] / parsed["n_units"]) if parsed["n_units"] else None,
                    "class_correct": parsed["predicted_class"] == (prompts.UNKNOWN if look["flagged"]
                                                                   else rec["label_true"]),
                    "technique_ok": (not parsed["mitre_ids"]) if look["flagged"]
                    else (bool(parsed["mitre_ids"]) and set(parsed["mitre_ids"]) <= tech_ok),
                    "mitigations_valid": all(c in pr["allowed_ids"] for c in parsed["mitigations_chosen"]),
                }
                _write(os.path.join(d, "output", "report.md"), raw)
                _write(os.path.join(d, "output", "report.json"),
                       {**parsed, "score": score, "confidence_logprob": conf})
                rec.update({"predicted_class": parsed["predicted_class"], "mitre_id": parsed["mitre_id"],
                            "assessment": parsed["assessment"], "n_sections": parsed["n_sections"],
                            "parse_error": parsed["parse_error"], "confidence_logprob": conf,
                            "vlm_generate_s": t_gen, **score})
                append_jsonl({"sample_id": sid, "user_input": pr["user"],
                              "retrieved_contexts": [c["text"] for c in pr["contexts"]],
                              "context_ids": [c["id"] for c in pr["contexts"]], "response": raw,
                              "system_prompt": pr["system"]}, os.path.join(out_root, "ragas.jsonl"))
    except Exception as e:                                   # ghi loi va chay tiep sample sau
        rec["run_error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-2000:]

    rec["latency"] = t.as_dict()
    rec["latency_total"] = t.total()
    _write(os.path.join(d, "record.json"), rec)
    append_jsonl(rec, os.path.join(out_root, "results.jsonl"))
    _latency_row(out_root, rec)
    return rec
