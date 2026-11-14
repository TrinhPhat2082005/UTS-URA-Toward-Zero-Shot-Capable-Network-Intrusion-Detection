"""
Build knowledge base: mapping/ + definitions/ + nguon da khoa  ->  store/passages.jsonl, index.json, build_report.md

    python LAYER2/knowledge_base/build/fetch_sources.py   # 1. tai + khoa nguon vao kb_sources/raw (mot lan)
    python LAYER2/knowledge_base/build/build_kb.py        # 2. sinh doan + chi muc + bao cao + validate

Moi phan cua KB la mot file part_*.py voi ham build(ctx) -> [passage]. Thu tu: M+B, H1, H2, H3, G.
Khong co so lieu nao tinh tu dataset.
"""
from __future__ import annotations

import datetime as dt
import json
import sys

import common as C
import mitre
import part_artefact
import part_feature
import part_mitigation
import part_protocol
import part_technique
import validate
from flag_order import flag_column_holds
from part_technique import technique_ids

PARTS = [("M+B", part_technique), ("H1", part_feature), ("H2", part_protocol), ("H3", part_artefact),
         ("G", part_mitigation)]
PART_NAMES = {"M": "M  label -> technique (Borisenko et al. 2022, Table II)",
              "B": "B  technique knowledge (ATT&CK v20.1, CAPEC v3.9)",
              "H1": "H1 feature definitions (CICFlowMeter-V3 2018)",
              "H2": "H2 protocol / port / TCP flag (IANA, IETF)",
              "H3": "H3 conditional extractor artefacts",
              "G": "G  mitigation candidates (ATT&CK mitigates + CAPEC Mitigations), chosen by the VLM"}


def load_ctx():
    d = C.DEFINITIONS_DIR
    return {"mapping": C.load_yaml(C.MAPPING_DIR / "borisenko2022_table2.yaml"),
            "feature_names": C.load_yaml(d / "feature_names.yaml"),
            "feature_notes": C.load_yaml(d / "feature_notes.yaml"),
            "model_features": C.load_yaml(d / "model_features.yaml"),
            "artefacts": C.load_yaml(d / "artefacts.yaml")}


def build_index(ctx, passages, kb_sha):
    """Chi muc nho cho retrieve.py (luc chay khong can yaml)."""
    pids = {p["pid"] for p in passages}
    labels = {}
    for label, tids in technique_ids(ctx["mapping"]).items():
        labels[label] = {
            "entries": ctx["mapping"]["labels"][label]["entries"],
            "techniques": tids,
            "capec": {t: [c["id"] for c in mitre.capec_linked(t)] for t in tids},
            "detection": {t: f"B-DET-{t}" in pids for t in tids},
        }
    return {"kb_sha256": kb_sha, "labels": labels, "flag_column_holds": flag_column_holds(),
            # cot roi rac (cong, giao thuc, co, Init Win): contrast ghi tan suat, khong lay trung binh
            "discrete_features": [f["csv"] for f in ctx["feature_names"] if f["discrete"]]}


# ----------------------------------------------------------------------------------------------
# Bao cao de duyet
# ----------------------------------------------------------------------------------------------
def _ev(e):
    return f"[{e['src']} | {e.get('locator', '')}] \"{e['quote']}\""


def report(passages, ctx, index, errors):
    lock, srcs = C.load_lock(), C.load_sources()
    L = ["# Knowledge base build report", "",
         f"Built {dt.date.today().isoformat()} | passages.jsonl sha256 `{index['kb_sha256']}`", "",
         "Validation: " + ("**passed** (every quote verbatim in its locked source, Borisenko key matches, coverage complete)"
                           if not errors else f"**{len(errors)} errors**"), ""]
    L += [f"- {e}" for e in errors]

    L += ["", "## 1. Sources", "", "| id | tier | version | retrieved | files (sha256) |", "|---|---|---|---|---|"]
    for sid, s in srcs.items():
        files = "; ".join(f"{n} `{v['sha256'][:12]}`" for n, v in lock.get(sid, {}).get("files", {}).items())
        L.append(f"| {sid} | {s['tier']} | {s['version']} | {lock.get(sid, {}).get('retrieved', '')} | {files} |")

    L += ["", "## 2. Passages per part", "", "| part | passages | verbatim | rendered | defined | sent to VLM |",
          "|---|---|---|---|---|---|"]
    for part in C.PART_ORDER:
        ps = [p for p in passages if p["part"] == part]
        k = {kind: sum(p["kind"] == kind for p in ps) for kind in sorted(validate.KINDS)}
        L.append(f"| {PART_NAMES[part]} | {len(ps)} | {k['verbatim']} | {k['rendered']} | {k['defined']} | "
                 f"{'no' if ps and not ps[0]['for_vlm'] else 'yes'} |")

    L += ["", "## 3. Label -> technique key (Borisenko et al. 2022, Table II)", "",
          "| label | Table II row (verbatim) | technique(s) used | CAPEC linked | network analytics |",
          "|---|---|---|---|---|"]
    for label, v in index["labels"].items():
        row = ctx["mapping"]["labels"][label]["row"]
        tids = ", ".join(f"{t} {mitre.technique(t)['name']}" for t in v["techniques"])
        cap = "; ".join(f"{t}: {', '.join(c) or 'none'}" for t, c in v["capec"].items())
        det = "; ".join(f"{t}: {'yes' if d else '**none**'}" for t, d in v["detection"].items())
        L.append(f"| {label} | {row['quote']} | {tids} | {cap} | {det} |")

    gaps = sorted({t for v in index["labels"].values() for t, d in v["detection"].items() if not d})
    L += ["", "### Gaps", "",
          "Techniques without any ATT&CK v20.1 analytic on a network data component (no B-DET passage): "
          + (", ".join(f"{t} {mitre.technique(t)['name']}" for t in gaps) or "none") + ".", ""]

    L += ["## 4. Defined sentences next to their evidence", "",
          "Code checks that each quote exists verbatim; a reviewer checks that the sentence says no more than its evidence.", ""]
    for p in passages:
        if p["kind"] == "defined":
            L += [f"### {p['pid']}", "", f"> {p['text']}", ""]
            L += [f"- {_ev(e)}" for e in p["support"]]
            L.append("")
    L += ["### Notes attached to H1 passages", ""]
    seen = set()
    for p in passages:
        for n in p["notes"]:
            key = (n["id"], n["text"])
            if key in seen:
                continue
            seen.add(key)
            cols = [q["keys"].get("feature", q["pid"]) for q in passages
                    if any((m["id"], m["text"]) == key for m in q["notes"])]
            L += [f"**{n['id']}** (on: {', '.join(cols)})", "", f"> {n['text']}", ""]
            L += [f"- {_ev(e)}" for e in n["support"]]
            L.append("")

    L += ["## 5. Flag columns (CICFlowMeter-V3 2018, Java HashMap order)", "",
          "| CSV column | flag it actually holds (first packet only) |", "|---|---|"]
    L += [f"| {col} | {held} |" for col, held in index["flag_column_holds"].items()]

    L += ["", "## 6. Short quotations (fewer than five words) for manual review", "",
          "Each is verbatim in its locked source (and in the file named by its locator).", "",
          "| source | locator | quote |", "|---|---|---|"]
    L += [f"| {src} | {loc} | `{q}` |" for (src, q), loc in sorted(validate.SHORT_QUOTES.items())]
    return "\n".join(L) + "\n"


def main():
    ctx = load_ctx()
    passages = []
    for name, mod in PARTS:
        ps = mod.build(ctx)
        print(f"  {name:<4} {len(ps):>5} doan  ({mod.__name__})")
        passages += ps
    C.PASSAGES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(C.PASSAGES_JSONL, "w", encoding="utf-8", newline="\n") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    kb_sha = C.sha256_file(C.PASSAGES_JSONL)
    index = build_index(ctx, passages, kb_sha)
    C.write_json(C.STORE_DIR / "index.json", index)
    errors = validate.run()
    (C.STORE_DIR / "build_report.md").write_text(report(passages, ctx, index, errors), encoding="utf-8", newline="\n")
    print(f"-> {C.PASSAGES_JSONL} ({len(passages)} doan, sha256 {kb_sha[:16]})")
    print(f"-> {C.STORE_DIR / 'build_report.md'}")
    for e in errors:
        print("  LOI", e)
    print("validate:", "OK" if not errors else f"{len(errors)} loi")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
