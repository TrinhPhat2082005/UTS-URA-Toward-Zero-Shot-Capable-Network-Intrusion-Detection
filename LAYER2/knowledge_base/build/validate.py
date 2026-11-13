"""
Kiem knowledge base. Chay doc lap hoac duoc build_kb.py goi o cuoi.

    python LAYER2/knowledge_base/build/validate.py              # kiem authored/*.yaml + store/passages.jsonl
    python LAYER2/knowledge_base/build/validate.py --authored   # chi kiem authored/*.yaml

Kiem gi:
  1. Moi file nguon khop sha256 trong sources.lock.json.
  2. Moi bang chung {src, quote} trong authored/*.yaml va passages.jsonl: quote nam NGUYEN VAN trong nguon
     (sau normalize), src co trong sources.yaml.
  3. Moi bang chung {data: khoa}: khoa ton tai trong data_facts.json.
  4. Moi quyet dinh {decision}: co it nhat mot support va co rationale.
  5. passages.jsonl: pid duy nhat; kind thuoc {verbatim, rendered, authored}; doan `verbatim` co text nam
     nguyen van trong nguon cua no; doan `rendered`/`authored` co it nhat mot support; do phu (moi nhan
     co A va B, moi cot co H1, moi cong chinh co H2).
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import common as C

KINDS = {"verbatim", "rendered", "authored"}

# Luat moi 2026-11-13:
#   D1   doan gui VLM chi duoc dung so lieu {data: claims.<id>} co holds_all = true (dung tren train, val va moi
#        file CSV goc). Ngoai le DERIVED_KEYS: ket qua do code tinh lai tu ma nguon (thu tu HashMap), khong
#        phai so lieu dataset (checklist 4.3, ban nhe).
#   4.6  khong co phan tram trong `text` gui VLM va trong `text`/`rationale` cua authored/*.yaml; chi cho phep
#        "99.9%" (nguong cua claim bat bien gan dung, vd near_identical).
DERIVED_KEYS = {"flag_column_holds"}

# checklist 5.2 (ban nhe): trich dan < 5 tu (cau lenh Java ngan, ten nhan trong Table 2, dong IANA) van hop le
# neu dung nguyen van, nhung duoc liet ke trong build_report.md de nguoi duyet doc lai.
SHORT_QUOTES = {}          # {(src, quote): locator dau tien gap}
PERCENT = re.compile(r"\d+(?:[.,]\d+)?\s*%")
PERCENT_ALLOWED = {"99.9%"}


def _percent_errors(text, where, errors):
    for m in PERCENT.findall(text or ""):
        if m.replace(" ", "") not in PERCENT_ALLOWED:
            errors.append(f"{where}: co phan tram '{m}' (luat 4.6: dung claim va cach noi khong co %)")


def _walk(node, path=""):
    """Duyet de quy, tra (duong dan, dict) cho moi dict."""
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def check_evidence(node, where, facts, sources, errors):
    for path, d in _walk(node):
        if "quote" in d and "src" in d:
            if d["src"] not in sources:
                errors.append(f"{where}:{path}: nguon '{d['src']}' khong co trong sources.yaml")
            elif not C.quote_ok(d["src"], d["quote"], d.get("locator")):
                scope = C.locator_file(d["src"], d.get("locator")) or d["src"]
                errors.append(f"{where}:{path}: quote KHONG nam nguyen van trong {scope}: {d['quote'][:90]!r}")
            elif len(d["quote"].split()) < 5:
                SHORT_QUOTES.setdefault((d["src"], d["quote"]), d.get("locator", ""))
        if "data" in d and isinstance(d["data"], str):
            try:
                v = C.fact(facts, d["data"])
                if d["data"].startswith("claims.") and not v.get("holds_all"):
                    errors.append(f"{where}:{path}: claim {d['data']} KHONG dung o moi nguon (failing={v.get('failing')})")
            except KeyError as e:
                errors.append(f"{where}:{path}: {e}")
        if "decision" in d:
            if not d.get("support"):
                errors.append(f"{where}:{path}: quyet dinh '{d['decision']}' khong co support")
            # checklist 5.4: `route` chi la cach kiem lien ket (part_mechanism.py), khong thay duoc ly do
            if not d.get("rationale"):
                errors.append(f"{where}:{path}: quyet dinh '{d['decision']}' khong co rationale")


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


def check_authored(facts, sources, errors):
    for path in sorted(C.AUTHORED_DIR.glob("*.yaml")):
        doc = C.load_yaml(path)
        check_evidence(doc, path.name, facts, sources, errors)
        for p, d in _walk(doc):
            for k in ("text", "rationale"):
                if isinstance(d.get(k), str):
                    _percent_errors(d[k], f"{path.name}:{p}.{k}", errors)


def _data_keys(node):
    return [d["data"] for _, d in _walk(node) if isinstance(d.get("data"), str)]


def check_vlm_stationary(p, errors):
    """D1: so lieu trong support/ghi chu cua doan gui VLM phai la claim dung o moi nguon."""
    if not p.get("for_vlm", True):
        return
    for key in _data_keys({"support": p.get("support"), "notes": p.get("notes")}):
        if key in DERIVED_KEYS or key.startswith("claims."):
            continue                                   # claims.*: holds_all da kiem trong check_evidence
        errors.append(f"passage {p['pid']}: so lieu '{key}' khong phai claim co kiem tinh dung (luat D1)")
    _percent_errors(p.get("text"), f"passage {p['pid']}", errors)
    _percent_errors(p.get("technique_text"), f"passage {p['pid']} (technique_text)", errors)
    for n in p.get("notes", []):
        _percent_errors(n.get("text"), f"passage {p['pid']} note {n.get('id')}", errors)


def check_passages(facts, sources, errors):
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
            src = p.get("source", {})
            if not C.quote_ok(src.get("src", ""), p["text"], src.get("locator")):
                errors.append(f"passage {pid}: text 'verbatim' khong nam nguyen van trong {src.get('src')}")
        elif not p.get("support"):
            errors.append(f"passage {pid}: doan {p.get('kind')} khong co support")
        check_evidence(p, f"passage {pid}", facts, sources, errors)
        check_vlm_stationary(p, errors)
    return ps


def check_coverage(ps, errors):
    pids = {p["pid"] for p in ps}
    mapping = C.load_yaml(C.AUTHORED_DIR / "mapping.yaml")
    for label, m in mapping.items():
        if f"A-{label}" not in pids:
            errors.append(f"do phu: nhan {label} khong co doan A")
        if f"B-{m['technique']['decision']}" not in pids:
            errors.append(f"do phu: technique {m['technique']['decision']} ({label}) khong co doan B")
        for ex in m.get("mechanism_extra", []):
            if f"B-{ex['decision']}" not in pids:
                errors.append(f"do phu: technique phu {ex['decision']} ({label}) khong co doan B")
    for f in C.load_yaml(C.AUTHORED_DIR / "feature_names.yaml"):
        if f"H1-{C.pid_safe(f['csv'])}" not in pids:
            errors.append(f"do phu: cot {f['csv']} khong co doan H1")
    for port in C.load_facts()["attack_main_ports"]["ports"]:
        # cong 0 = gia tri mac dinh cua cong cu (khong phai cong IANA): do H3-NOTRANSPORT giai thich
        want = "H3-NOTRANSPORT" if port == "0" else f"H2-PORT-{port}"
        if want not in pids:
            errors.append(f"do phu: cong chinh {port} khong co doan {want}")


def run(authored_only=False):
    errors = []
    SHORT_QUOTES.clear()
    facts, sources = C.load_facts(), C.load_sources()
    check_lock(errors)
    check_authored(facts, sources, errors)
    if not authored_only:
        ps = check_passages(facts, sources, errors)
        if ps:
            check_coverage(ps, errors)
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authored", action="store_true")
    args = ap.parse_args()
    errors = run(args.authored)
    for e in errors:
        print("  LOI", e)
    print(f"\n{len(errors)} loi" if errors else "\nKB hop le: moi trich dan nguyen van, moi so lieu co khoa, do phu du")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
