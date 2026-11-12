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
import sys

import common as C

KINDS = {"verbatim", "rendered", "authored"}


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
            elif not C.quote_ok(d["src"], d["quote"]):
                errors.append(f"{where}:{path}: quote KHONG nam nguyen van trong {d['src']}: {d['quote'][:90]!r}")
        if "data" in d and isinstance(d["data"], str):
            try:
                C.fact(facts, d["data"])
            except KeyError as e:
                errors.append(f"{where}:{path}: {e}")
        if "decision" in d:
            if not d.get("support"):
                errors.append(f"{where}:{path}: quyet dinh '{d['decision']}' khong co support")
            if not d.get("rationale") and not d.get("route"):
                errors.append(f"{where}:{path}: quyet dinh '{d['decision']}' khong co rationale/route")


def check_lock(errors):
    lock = C.load_lock()
    for sid, spec in C.load_sources().items():
        for f in spec.get("files", []) + spec.get("local", []):
            p = C.source_path(sid, f["name"])
            want = lock.get(sid, {}).get("files", {}).get(f["name"], {}).get("sha256")
            if not p.exists():
                errors.append(f"lock: thieu file {p}")
            elif C.sha256_file(p) != want:
                errors.append(f"lock: {sid}/{f['name']} khong khop sha256 trong sources.lock.json")


def check_authored(facts, sources, errors):
    for path in sorted(C.AUTHORED_DIR.glob("*.yaml")):
        check_evidence(C.load_yaml(path), path.name, facts, sources, errors)


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
            if not C.quote_ok(src.get("src", ""), p["text"]):
                errors.append(f"passage {pid}: text 'verbatim' khong nam nguyen van trong {src.get('src')}")
        elif not p.get("support"):
            errors.append(f"passage {pid}: doan {p.get('kind')} khong co support")
        check_evidence(p, f"passage {pid}", facts, sources, errors)
    return ps


def check_coverage(ps, errors):
    pids = {p["pid"] for p in ps}
    mapping = C.load_yaml(C.AUTHORED_DIR / "mapping.yaml")
    for label, m in mapping.items():
        if f"A-{label}" not in pids:
            errors.append(f"do phu: nhan {label} khong co doan A")
        if f"B-{m['technique']['decision']}" not in pids:
            errors.append(f"do phu: technique {m['technique']['decision']} ({label}) khong co doan B")
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
