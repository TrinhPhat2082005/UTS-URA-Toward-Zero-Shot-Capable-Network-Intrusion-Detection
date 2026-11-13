"""
Ham dung chung cho viec build knowledge base (KB).

Ba loai noi dung duoc phep trong KB (xem knowledge_base/README.md):
    trich dan   {"src": <id nguon>, "locator": <vi tri>, "quote": <cau nguyen van>}
    so lieu     {"data": <khoa trong store/data_facts.json>}
    quyet dinh  {"decision": <gia tri>, "support": [<trich dan | so lieu>, ...], "rationale": <ly do>}

File nay chi lo: duong dan, doc nguon da khoa sha256, chuan hoa van ban de so trich dan, doc data_facts.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml

KB_DIR = Path(__file__).resolve().parents[1]          # LAYER2/knowledge_base
LAYER2_DIR = KB_DIR.parent
REPO_DIR = LAYER2_DIR.parent                          # UTS_URA
AUTHORED_DIR = KB_DIR / "authored"
STORE_DIR = KB_DIR / "store"
SOURCES_YAML = Path(__file__).resolve().parent / "sources.yaml"
LOCK_JSON = STORE_DIR / "sources.lock.json"
FACTS_JSON = STORE_DIR / "data_facts.json"
PASSAGES_JSONL = STORE_DIR / "passages.jsonl"

if str(LAYER2_DIR) not in sys.path:
    sys.path.insert(0, str(LAYER2_DIR))
import config  # noqa: E402

# File nguon tai ve nam cung du lieu nang ($UTS_DATA), khong commit vao repo.
SOURCES_DIR = Path(config.DATA_ROOT) / "kb_sources"


# ----------------------------------------------------------------------------------------------
# Doc file cau hinh
# ----------------------------------------------------------------------------------------------
def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_sources():
    """{id: muc trong sources.yaml}."""
    return {s["id"]: s for s in load_yaml(SOURCES_YAML)}


def load_lock():
    return json.loads(LOCK_JSON.read_text(encoding="utf-8")) if LOCK_JSON.exists() else {}


CLAIMS_JSON = STORE_DIR / "claims.json"


def load_facts():
    """data_facts.json (mo ta, chi train) + claims.json (co kiem tinh dung, build/claims.py) duoi khoa 'claims'.
    Doan gui VLM chi duoc dung {data: claims.<id>} co holds_all = true (validate.py)."""
    facts = json.loads(FACTS_JSON.read_text(encoding="utf-8"))
    facts["claims"] = json.loads(CLAIMS_JSON.read_text(encoding="utf-8")) if CLAIMS_JSON.exists() else {}
    return facts


def claim_key(facts, a, b, kind="identical"):
    """Khoa claim cua cap cot (a, b) bat ke thu tu: 'claims.identical__a__b' hoac '...__b__a'."""
    for x, y in ((a, b), (b, a)):
        cid = f"{kind}__{x}__{y}"
        if cid in facts["claims"]:
            return f"claims.{cid}"
    raise KeyError(f"claims.json khong co claim {kind} cho cap ({a}, {b}): chay build/claims.py")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lf(path):
    """sha256 sau khi doi CRLF -> LF. Dung cho nguon `local` (file trong repo): git checkout tren Windows
    ghi CRLF, tren Linux ghi LF; bam noi dung LF thi lock giong nhau tren moi may (checklist 4.5)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def is_local(src_id, name):
    return any(item["name"] == name for item in load_sources()[src_id].get("local", []))


def sha256_source(src_id, name):
    """Hash dung trong lock: file tai ve -> sha256 nguyen byte; file trong repo -> sha256_lf."""
    p = source_path(src_id, name)
    return sha256_lf(p) if is_local(src_id, name) else sha256_file(p)


def source_path(src_id, name):
    """Duong dan file `name` cua nguon `src_id` (file tai ve, hoac file trong repo voi nguon local)."""
    spec = load_sources()[src_id]
    for item in spec.get("local", []):
        if item["name"] == name:
            return REPO_DIR / item["path"]
    return SOURCES_DIR / src_id / name


def source_files(src_id):
    spec = load_sources()[src_id]
    return [source_path(src_id, f["name"]) for f in spec.get("files", []) + spec.get("local", [])]


# ----------------------------------------------------------------------------------------------
# Van ban cua nguon (de so trich dan nguyen van)
# ----------------------------------------------------------------------------------------------
_CITATION = re.compile(r"\(Citation:[^)]*\)")
_MDLINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_TAGS = re.compile(r"<[^>]+>")


def normalize(text):
    """Chuan hoa de so nguyen van (ap dung NHU NHAU cho trich dan va nguon):
    NFKC; bo '(Citation: ...)' va link markdown '[x](url)' -> 'x' cua ATT&CK; gop khoang trang;
    bo khoang trang truoc dau cau (sinh ra khi go the HTML)."""
    text = unicodedata.normalize("NFKC", text)
    text = _CITATION.sub("", text)
    text = _MDLINK.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r" ([,.;:)])", r"\1", text).strip()


def _html_text(raw):
    raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S | re.I)
    return html.unescape(_TAGS.sub(" ", raw))


def _stix_text(path):
    """Moi truong description + name cua bundle STIX, noi lai."""
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    return "\n".join(f"{o.get('name', '')}\n{o.get('description', '')}" for o in bundle["objects"])


def _xml_text(path):
    import xml.etree.ElementTree as ET
    return " ".join(ET.parse(path).getroot().itertext())


@lru_cache(maxsize=None)
def source_text(src_id):
    """Toan bo van ban (da chuan hoa) cua mot nguon, dung de kiem trich dan."""
    parts = []
    for p in source_files(src_id):
        if not p.exists():
            raise FileNotFoundError(f"thieu file nguon {p}; chay build/fetch_sources.py truoc")
        if p.suffix == ".json":
            parts.append(_stix_text(p))
        elif p.suffix == ".xml":
            parts.append(_xml_text(p))
        elif p.suffix == ".html":
            parts.append(_html_text(p.read_text(encoding="utf-8", errors="replace")))
        else:
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return normalize("\n".join(parts))


def quote_ok(src_id, quote):
    """Trich dan co nam nguyen van trong nguon khong (sau khi chuan hoa)."""
    return normalize(quote) in source_text(src_id)


# ----------------------------------------------------------------------------------------------
# So lieu (store/data_facts.json)
# ----------------------------------------------------------------------------------------------
def fact(facts, key):
    """Lay gia tri theo khoa dang 'a.b.c'. Ten cot co the chua '/', khong chua '.'.
    Khoa 'claims.<id>' tra ca muc claim {statement, values, holds_all, ...}."""
    if key.startswith("claims."):
        cid = key[len("claims."):]
        if cid not in facts.get("claims", {}):
            raise KeyError(f"claims.json khong co claim '{cid}'")
        return facts["claims"][cid]
    node = facts
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"data_facts khong co khoa '{key}' (dung o '{part}')")
        node = node[part]
    return node


# ----------------------------------------------------------------------------------------------
# Mot doan van (passage) cua KB
# ----------------------------------------------------------------------------------------------
PART_ORDER = ["A", "B", "H1", "H2", "H3", "G"]


def passage(pid, part, keys, kind, text, *, title=None, source=None, support=None, notes=None,
            when=None, for_vlm=True):
    """Mot doan cua store/passages.jsonl.

    kind:
      verbatim  `text` la chuoi con nguyen van cua nguon `source` ({src, locator}); validate.py kiem.
      rendered  `text` do CODE ghep tu cac truong nguyen van trong `support` (vd ten cot + dinh nghia ReadMe,
                dong dang ky IANA); khong co cau nao do nguoi viet.
      authored  `text` do tac gia viet (nam trong authored/*.yaml), kem `support`.
    notes: [{id, text, support}] ghi chu tac gia gan them (H1), moi ghi chu co bang chung rieng.
    for_vlm: False voi G' (code dien vao report, khong dua cho VLM).
    """
    assert kind in ("verbatim", "rendered", "authored"), kind
    assert part in PART_ORDER, part
    if kind == "verbatim":
        assert source and "src" in source, f"{pid}: doan verbatim can source"
        text = normalize(text)
    return {"pid": pid, "part": part, "title": title, "keys": keys, "kind": kind, "text": text,
            "source": source, "support": support or [], "notes": notes or [], "when": when,
            "for_vlm": for_vlm}


def pid_safe(name):
    """Ten cot -> phan cua pid: 'Flow Byts/s' -> 'Flow_Byts_per_s'."""
    name = name.replace("/", "_per_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
