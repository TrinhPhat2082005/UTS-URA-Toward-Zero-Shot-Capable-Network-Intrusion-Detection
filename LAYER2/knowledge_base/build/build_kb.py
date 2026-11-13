"""
Build knowledge base: authored/*.yaml + nguon da khoa + data_facts  ->  store/passages.jsonl + build_report.md

    python LAYER2/knowledge_base/build/fetch_sources.py   # 1. tai + khoa nguon (mot lan)
    python LAYER2/knowledge_base/build/data_facts.py      # 2. so lieu dataset
    python LAYER2/knowledge_base/build/build_kb.py        # 3. sinh doan + bao cao + validate

Moi phan cua KB la mot file part_*.py voi ham build(ctx) -> [passage]. Thu tu: A, B, H1, H2, H3, G.
"""
from __future__ import annotations

import datetime as dt
import json
import sys

import common as C
import mitre
import part_artefact
import part_feature
import part_label
import part_mechanism
import part_mitigation
import part_protocol
import validate

PARTS = [("A", part_label), ("B", part_mechanism), ("H1", part_feature), ("H2", part_protocol),
         ("H3", part_artefact), ("G", part_mitigation)]
PART_NAMES = {"A": "A' label card", "B": "B' mechanism (ATT&CK + CAPEC)", "H1": "H1 feature definitions",
              "H2": "H2 protocol / port / TCP flag", "H3": "H3 conditional extractor artefacts",
              "G": "G' mitigations (code-filled, not sent to the VLM)"}


def load_ctx():
    a = C.AUTHORED_DIR
    return {"mapping": C.load_yaml(a / "mapping.yaml"),
            "feature_names": C.load_yaml(a / "feature_names.yaml"),
            "extra_features": C.load_yaml(a / "extra_features.yaml"),
            "extractor_notes": C.load_yaml(a / "extractor_notes.yaml"),
            "artefacts": C.load_yaml(a / "artefacts.yaml"),
            "mechanism": C.load_yaml(a / "mechanism.yaml"),
            "facts": C.load_facts()}


def legacy_techniques():
    """Technique trong profile cu (knowledge_base/attack_profiles), de so sanh trong bao cao."""
    from knowledge_base.profiles import LABEL_TO_PROFILE_ID, load_profiles
    prof = load_profiles()
    return {lab: prof[pid].get("mitre_id", "?") for lab, pid in LABEL_TO_PROFILE_ID.items()}


# ----------------------------------------------------------------------------------------------
# Bao cao de duyet
# ----------------------------------------------------------------------------------------------
def _ev(e):
    if "quote" in e:
        return f"[{e['src']} | {e.get('locator', '')}] \"{e['quote']}\""
    if "data" in e and e["data"].startswith("claims."):
        c = C.fact(C.load_facts(), e["data"])
        return (f"[claim | {e['data'][7:]}] {c['statement']} — holds on every source: "
                f"{'yes' if c['holds_all'] else '**NO**'} ({len([v for v in c['values'].values() if v is not None])} sources)")
    if "data" in e:
        return f"[data_facts | {e['data']}] = {json.dumps(C.fact(C.load_facts(), e['data']), ensure_ascii=False)}"
    if "decision" in e:
        return f"[decision {e['decision']}] {e.get('rationale', '')}"
    return json.dumps(e, ensure_ascii=False)


def report(passages, ctx, kb_sha, errors):
    facts, lock, srcs = ctx["facts"], C.load_lock(), C.load_sources()
    L = [f"# Knowledge base build report", "",
         f"Built {dt.date.today().isoformat()} | passages.jsonl sha256 `{kb_sha}` | "
         f"dataset `{facts['dataset']['file']}` sha256 `{facts['dataset']['sha256'][:16]}...`", "",
         "Validation: " + ("**passed** (every quote verbatim in its locked source, every data key present, coverage complete)"
                           if not errors else f"**{len(errors)} errors**"), ""]
    for e in errors:
        L.append(f"- {e}")

    L += ["", "## 1. Sources", "", "| id | tier | version | retrieved | files (sha256) |", "|---|---|---|---|---|"]
    for sid, s in srcs.items():
        files = "; ".join(f"{n} `{v['sha256'][:12]}`" for n, v in lock.get(sid, {}).get("files", {}).items())
        L.append(f"| {sid} | {s['tier']} | {s['version']} | {lock.get(sid, {}).get('retrieved', '')} | {files} |")

    L += ["", "## 2. Passages per part", "", "| part | passages | verbatim | rendered | authored | sent to VLM |",
          "|---|---|---|---|---|---|"]
    for part, _ in PARTS:
        ps = [p for p in passages if p["part"] == part]
        k = {kind: sum(p["kind"] == kind for p in ps) for kind in ("verbatim", "rendered", "authored")}
        L.append(f"| {PART_NAMES[part]} | {len(ps)} | {k['verbatim']} | {k['rendered']} | {k['authored']} | "
                 f"{'no' if ps and not ps[0]['for_vlm'] else 'yes'} |")

    legacy = legacy_techniques()
    L += ["", "## 3. Label mapping (the only author decisions)", "",
          "| label | technique | previous profile | changed | CAPEC (route) | CAPEC entries CAPEC links to the technique |",
          "|---|---|---|---|---|---|"]
    for label, m in ctx["mapping"].items():
        tid = m["technique"]["decision"]
        cap = m["capec"]
        linked = ", ".join(c["id"] for c in mitre.capec_linked(tid)) or "none"
        L.append(f"| {label} | {tid} {mitre.technique(tid)['name']} | {legacy.get(label, '?')} | "
                 f"{'**yes**' if legacy.get(label) != tid else 'no'} | {cap['decision'] or '-'} ({cap['route']}) | {linked} |")
    L += ["", "Rationale for each technique and CAPEC choice:", ""]
    for label, m in ctx["mapping"].items():
        L.append(f"- **{label}** — {m['technique']['rationale']} CAPEC: {m['capec'].get('rationale', '')}")

    L += ["", "## 4. Every author-written sentence next to its evidence", "",
          "Code checks that each quote exists verbatim; a reviewer checks that the sentence says no more than its evidence.", ""]
    for p in passages:
        if p["kind"] == "authored":
            L += [f"### {p['pid']}", "", f"> {p['text']}", ""]
            L += [f"- {_ev(e)}" for e in p["support"]]
            L.append("")
    L += ["### Notes attached to H1 and H2 passages", ""]
    seen = set()
    for p in passages:
        for n in p["notes"]:
            key = (n["id"], n["text"])
            if key in seen:
                continue
            seen.add(key)
            # H1 co khoa feature; H2-PORT (ghi chu RFC 6335, checklist 1.5d) co khoa port -> dung pid
            cols = [q["keys"].get("feature", q["pid"]) for q in passages
                    if any((m["id"], m["text"]) == key for m in q["notes"])]
            L += [f"**{n['id']}** (on: {', '.join(cols)})", "", f"> {n['text']}", ""]
            L += [f"- {_ev(e)}" for e in n["support"]]
            L.append("")

    L += ["## 5. H3 conditions: share of flows per label (train_full.csv)", "",
          "Descriptive only: training split, not checked for stationarity, not used as evidence in any passage.", ""]
    arts = [a for a in ctx["artefacts"] if a.get("frequency_key")]
    L.append("| label | " + " | ".join(a["pid"] for a in arts) + " |")
    L.append("|---|" + "---|" * len(arts))
    for label, v in facts["by_label"].items():
        L.append(f"| {label} | " + " | ".join(f"{v[a['frequency_key']]:.1%}" for a in arts) + " |")

    L += ["", "## 6. Findings from the 2018 extractor source code", "",
          "Flag columns (each holds the flag of the FIRST packet only):", "",
          "| CSV column | flag it actually holds |", "|---|---|"]
    for col, held in facts["flag_column_holds"].items():
        L.append(f"| {col} | {held} |")
    L += ["", f"Consistency in the data: {json.dumps(facts['flag_consistency'])}", "",
          f"Constant columns: {', '.join(facts['constant_columns'])}", "",
          f"Identical column pairs: {facts['identical_columns']}; near-identical: {facts['near_identical_columns']}", ""]

    # checklist 4.1: khoang phien ban cua extractor (khong chung minh duoc commit nao sinh dataset)
    L += ["## 6a. Extractor version bracket", "",
          "No commit can be shown to have produced the published CSV files. The data are consistent with the "
          "CICFlowMeter-V3 code from eaa853dd (2018-02-13) to 50d668db (2018-03-07), whose BasicFlow.java is "
          "identical, and inconsistent with later code:", "",
          "- flags only from the first packet: claim `flags_binary` (code commented out in addPacket(); counting "
          "for every packet came back only in ed1d6bcb, 2019-10-15);",
          "- first packet counted twice in the length statistics: claim `pkt_len_double_count`;",
          "- integer Down/Up Ratio: claim `downup_integer`;",
          "- Timestamp written as 12-hour time without AM/PM, matching `format = \"dd/MM/yyyy hh:mm:ss\";` in "
          "DateFormatter.java at 50d668db; commit 21790da0 (2018-05-31) added AM/PM;",
          "- flag columns in Java HashMap order; commit 6833ccd4 (2018-06-04) writes them in an explicit order "
          "(source `cicflowmeter-src-2018-06`).", ""]

    # checklist 5.2: trich dan ngan (< 5 tu) hop le nhung can nguoi duyet doc lai
    L += ["## 6c. Short quotations (fewer than five words) for manual review", "",
          "Each is verbatim in its locked source (and in the file named by its locator); short Java statements, "
          "label names of Table 2 and IANA rows are expected here.", "", "| source | locator | quote |", "|---|---|---|"]
    for (src, q), loc in sorted(validate.SHORT_QUOTES.items()):
        L.append(f"| {src} | {loc} | `{q}` |")
    L.append("")

    # checklist 1.1: cot ma dinh nghia ReadMe khong dung voi dataset nay (khong con trong text gui VLM)
    L += ["## 6b. ReadMe definitions that do not hold for CSE-CIC-IDS2018", "",
          "These columns start with their real meaning; the ReadMe definition is kept only in "
          "authored/feature_names.yaml and in the passage support.", "", "| column | ReadMe definition (not sent to the VLM) |",
          "|---|---|"]
    readme = {f["csv"]: f.get("definition", "") for f in ctx["feature_names"]}
    for p in passages:
        if p.get("readme_replaced"):
            L.append(f"| {p['keys']['feature']} | {readme.get(p['keys']['feature'], '')} |")

    # checklist D1: claim so lieu va tinh dung
    L += ["", "## 7. Claims and stationarity (build/claims.py)", "",
          "A number enters a passage sent to the VLM only as a claim that holds on the training split, the validation "
          "split and every published CSV file (and both halves of each label within a file, by row order).", "",
          "| claim | statement | test | train | val | min / max over raw files and halves | holds |",
          "|---|---|---|---|---|---|---|"]
    for cid, c in facts["claims"].items():
        raw = [v for k, v in c["values"].items() if k not in ("train", "val") and v is not None]
        rng = f"{min(raw):.4g} / {max(raw):.4g}" if raw else "-"
        fmt = lambda v: "-" if v is None else f"{v:.4g}"
        L.append(f"| {cid} | {c['statement']} | {c['test']} | {fmt(c['values'].get('train'))} | "
                 f"{fmt(c['values'].get('val'))} | {rng} | {'yes' if c['holds_all'] else '**no**'} |")
    return "\n".join(L) + "\n"


def main():
    ctx = load_ctx()
    passages = []
    for part, mod in PARTS:
        ps = mod.build(ctx)
        print(f"  {part:<3} {len(ps):>4} doan  ({mod.__name__})")
        passages += ps
    C.PASSAGES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(C.PASSAGES_JSONL, "w", encoding="utf-8", newline="\n") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    kb_sha = C.sha256_file(C.PASSAGES_JSONL)
    # Chi muc nho cho retrieve.py (luc chay khong can yaml): nhan -> technique/CAPEC; cot co -> co thuc chua.
    C.write_json(C.STORE_DIR / "index.json", {
        "kb_sha256": kb_sha,
        "labels": {lab: {"technique": m["technique"]["decision"], "capec": m["capec"]["decision"],
                         "extra_mechanisms": [ex["decision"] for ex in m.get("mechanism_extra", [])]}
                   for lab, m in ctx["mapping"].items()},
        "flag_column_holds": {**ctx["facts"]["flag_column_holds"], "Fwd PSH Flags": "PSH", "Fwd URG Flags": "URG"},
        # cot roi rac (cong, giao thuc, co, Init Win co gia tri thay the -1): bang C ghi tan suat, khong lay trung binh
        "discrete_features": [f["csv"] for f in ctx["feature_names"] if f["discrete"]],
    })
    errors = validate.run()
    (C.STORE_DIR / "build_report.md").write_text(report(passages, ctx, kb_sha, errors), encoding="utf-8")
    print(f"-> {C.PASSAGES_JSONL} ({len(passages)} doan, sha256 {kb_sha[:16]})")
    print(f"-> {C.STORE_DIR / 'build_report.md'}")
    for e in errors:
        print("  LOI", e)
    print("validate:", "OK" if not errors else f"{len(errors)} loi")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
