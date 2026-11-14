"""
Kiem knowledge base. Chay doc lap hoac duoc build_kb.py goi o cuoi.

    python LAYER2/knowledge_base/build/validate.py               # kiem mapping/, definitions/ + store/passages.jsonl
    python LAYER2/knowledge_base/build/validate.py --definitions # chi kiem mapping/ va definitions/

Kiem gi:
  1. Moi file nguon (ca file dat tay) khop sha256 trong sources.lock.json.
  2. Moi trich dan {src, locator, quote} trong mapping/, definitions/ va passages.jsonl: src co trong sources.yaml,
     quote nam NGUYEN VAN trong nguon (trong file ma locator chi toi, neu co).
  3. Khong con bang chung kieu cu: khong co khoa `data` (so lieu tu tinh) hay `decision` (quyet dinh tac gia).
  4. Khoa Borisenko: du 14 nhan; moi ID (tactic, technique, sub-technique) cua `entries` nam trong dong Table II
     da trich (`row.quote`); moi ID ton tai trong ATT&CK v20.1.
  5. Khong co phan tram trong cau tu dinh nghia (doan `defined`, ghi chu H1, definitions/*.yaml).
  6. passages.jsonl: pid duy nhat; kind thuoc {verbatim, rendered, defined}; doan `verbatim` co text nguyen van
     trong nguon cua no; doan khac co it nhat mot support.
  7. Do phu: M-<nhan> moi nhan; B-<technique> moi technique; H1 moi cot CSV va moi feature mo hinh;
     H2-PROTO-6/17, H2-PORT-NOTE, H2-FLAG-<co> cho 8 co; H3 moi artefact.
  8. Dieu kien H3-WINSCALE dung cot 'PSH Flag Cnt' vi cot nay chua SYN (flag_order.py).
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import common as C
import mitre
from flag_order import FLAG_KEYS, flag_column_holds
from part_technique import technique_ids

KINDS = {"verbatim", "rendered", "defined"}
FORBIDDEN_KEYS = ("data", "decision", "frequency_key")
PERCENT = re.compile(r"\d+(?:[.,]\d+)?\s*%")
ID_RE = re.compile(r"^(TA\d{4}|T\d{4}(?:\.\d{3})?)$")

# Trich dan < 5 tu (cau lenh Java ngan, dong IANA) van hop le neu nguyen van; build_report.md liet ke de doc lai.
SHORT_QUOTES = {}          # {(src, quote): locator dau tien gap}
REGISTRY_SOURCES = {"iana-ports", "iana-protocols"}   # dong registry do code chep nguyen dong, khong can doc tay


def _walk(node, path=""):
    """Duyet de quy, tra (duong dan, dict) cho moi dict."""
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def _percent(text, where, errors):
    for m in PERCENT.findall(text or ""):
        errors.append(f"{where}: co phan tram '{m}' (cau tu dinh nghia khong duoc chua so lieu)")


def check_evidence(node, where, sources, errors):
    for path, d in _walk(node):
        for k in FORBIDDEN_KEYS:
            if k in d:
                errors.append(f"{where}:{path}: khoa '{k}' khong con hop le (KB chi dung trich dan nguon)")
        if "quote" in d:                       # {src, locator} khong co quote = tham chieu, khong phai trich dan
            if "src" not in d:
                errors.append(f"{where}:{path}: trich dan thieu src")
            elif d["src"] not in sources:
                errors.append(f"{where}:{path}: nguon '{d['src']}' khong co trong sources.yaml")
            elif not C.quote_ok(d["src"], d["quote"], d.get("locator")):
                scope = C.locator_file(d["src"], d.get("locator")) or d["src"]
                errors.append(f"{where}:{path}: quote KHONG nam nguyen van trong {scope}: {d['quote'][:90]!r}")
            elif len(d["quote"].split()) < 5 and d["src"] not in REGISTRY_SOURCES:
                SHORT_QUOTES.setdefault((d["src"], d["quote"]), d.get("locator", ""))


def check_lock(errors):
    lock = C.load_lock()
    for sid, spec in C.load_sources().items():
        for f in spec.get("files", []) + spec.get("local", []):
            p = C.source_path(sid, f["name"])
            want = lock.get(sid, {}).get("files", {}).get(f["name"], {}).get("sha256")
            if not p.exists():
                errors.append(f"lock: thieu file {p}")
            elif C.sha256_source(sid, f["name"]) != want:
                errors.append(f"lock: {sid}/{f['name']} khong khop sha256 trong sources.lock.json")


def check_mapping(table, errors):
    labels = table.get("labels", {})
    if len(labels) != 14:
        errors.append(f"mapping: Table II co 14 nhan, file co {len(labels)}")
    for label, row in labels.items():
        quote = row["row"]["quote"]
        if not row.get("entries"):
            errors.append(f"mapping {label}: khong co entry")
        for e in row.get("entries", []):
            for k in ("tactic", "technique", "sub_technique"):
                v = e.get(k)
                if v is None:
                    if k != "sub_technique":
                        errors.append(f"mapping {label}: thieu {k}")
                    continue
                if not ID_RE.match(v):
                    errors.append(f"mapping {label}: {k} '{v}' khong phai ID ATT&CK")
                elif f"({v})" not in C.normalize(quote):
                    errors.append(f"mapping {label}: {k} {v} KHONG nam trong dong Table II da trich")
            try:
                mitre.tactic_name(e["tactic"])
                mitre.technique(e["technique"])
                if e.get("sub_technique"):
                    mitre.technique(e["sub_technique"])
            except KeyError as ex:
                errors.append(f"mapping {label}: ID khong co trong ATT&CK v20.1: {ex}")


def check_definitions(sources, errors):
    files = sorted(C.MAPPING_DIR.glob("*.yaml")) + sorted(C.DEFINITIONS_DIR.glob("*.yaml"))
    for path in files:
        doc = C.load_yaml(path)
        check_evidence(doc, path.name, sources, errors)
        if path.parent == C.DEFINITIONS_DIR and path.name != "feature_names.yaml":
            for p, d in _walk(doc):
                if isinstance(d.get("text"), str):
                    _percent(d["text"], f"{path.name}:{p}.text", errors)
    check_mapping(C.load_yaml(C.MAPPING_DIR / "borisenko2022_table2.yaml"), errors)
    # H3-WINSCALE: "goi dau khong phai SYN" doc tu cot chua SYN cua goi dau
    holds = flag_column_holds()
    for a in C.load_yaml(C.DEFINITIONS_DIR / "artefacts.yaml"):
        if a["pid"] == "H3-WINSCALE" and holds.get("PSH Flag Cnt") != "SYN":
            errors.append("H3-WINSCALE: cot 'PSH Flag Cnt' khong con chua SYN theo flag_order.py")


def check_passages(sources, errors):
    if not C.PASSAGES_JSONL.exists():
        errors.append("thieu store/passages.jsonl (chay build_kb.py)")
        return []
    ps = [json.loads(line) for line in C.PASSAGES_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    for p in ps:
        pid = p.get("pid")
        if pid in seen:
            errors.append(f"passage {pid}: pid trung")
        seen.add(pid)
        if p.get("kind") not in KINDS:
            errors.append(f"passage {pid}: kind '{p.get('kind')}' khong hop le")
        if not p.get("text"):
            errors.append(f"passage {pid}: text rong")
        if p.get("kind") == "verbatim":
            src = p.get("source") or {}
            if not C.quote_ok(src.get("src", ""), p["text"], src.get("locator")):
                errors.append(f"passage {pid}: text 'verbatim' khong nam nguyen van trong {src.get('src')}")
        elif not p.get("support"):
            errors.append(f"passage {pid}: doan {p.get('kind')} khong co support")
        if p.get("kind") == "defined":
            _percent(p.get("text"), f"passage {pid}", errors)
        for n in p.get("notes", []):
            _percent(n.get("text"), f"passage {pid} note {n.get('id')}", errors)
        check_evidence({"support": p.get("support"), "notes": p.get("notes")}, f"passage {pid}", sources, errors)
    return ps


def check_coverage(ps, errors):
    pids = {p["pid"] for p in ps}
    want = []
    table = C.load_yaml(C.MAPPING_DIR / "borisenko2022_table2.yaml")
    for label, tids in technique_ids(table).items():
        want.append((f"M-{label}", f"nhan {label}"))
        want += [(f"B-{t}", f"technique {t} ({label})") for t in tids]
    want += [(f"H1-{C.pid_safe(f['csv'])}", f"cot {f['csv']}")
             for f in C.load_yaml(C.DEFINITIONS_DIR / "feature_names.yaml")]
    want += [(e["pid"], "feature mo hinh") for e in C.load_yaml(C.DEFINITIONS_DIR / "model_features.yaml")]
    want += [("H2-PROTO-6", "giao thuc TCP"), ("H2-PROTO-17", "giao thuc UDP"), ("H2-PORT-NOTE", "ghi chu cong")]
    want += [(f"H2-FLAG-{f}", f"co {f}") for f in FLAG_KEYS]
    want += [(a["pid"], "artefact") for a in C.load_yaml(C.DEFINITIONS_DIR / "artefacts.yaml")]
    for pid, what in want:
        if pid not in pids:
            errors.append(f"do phu: {what} khong co doan {pid}")


def run(definitions_only=False):
    errors = []
    SHORT_QUOTES.clear()
    sources = C.load_sources()
    check_lock(errors)
    check_definitions(sources, errors)
    if not definitions_only:
        ps = check_passages(sources, errors)
        if ps:
            check_coverage(ps, errors)
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--definitions", action="store_true")
    args = ap.parse_args()
    errors = run(args.definitions)
    for e in errors:
        print("  LOI", e)
    print(f"\n{len(errors)} loi" if errors else "\nKB hop le: moi trich dan nguyen van, khoa Borisenko khop, do phu du")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
