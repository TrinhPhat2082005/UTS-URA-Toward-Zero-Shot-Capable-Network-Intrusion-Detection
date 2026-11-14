"""
Ham dung chung cho viec build knowledge base (KB).

KB chi co MOT loai bang chung (xem knowledge_base/README.md):
    trich dan   {"src": <id nguon>, "locator": <vi tri>, "quote": <cau nguyen van>}   -> validate.py kiem nguyen van
Khong co quyet dinh cua tac gia, khong co so lieu tu tinh tu dataset.

File nay chi lo: duong dan, doc nguon da khoa sha256 (thu muc raw/), chuan hoa van ban de so trich dan,
dinh dang mot doan (passage).
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
MAPPING_DIR = KB_DIR / "mapping"                      # khoa nhan -> technique (Borisenko 2022, Table II)
DEFINITIONS_DIR = KB_DIR / "definitions"              # dinh nghia feature, co, feature AE, artefact H3
STORE_DIR = KB_DIR / "store"
SOURCES_YAML = Path(__file__).resolve().parent / "sources.yaml"
LOCK_JSON = STORE_DIR / "sources.lock.json"
PASSAGES_JSONL = STORE_DIR / "passages.jsonl"

if str(LAYER2_DIR) not in sys.path:
    sys.path.insert(0, str(LAYER2_DIR))
import config  # noqa: E402

# File nguon tai ve (hoac dat tay voi nguon `manual`) nam cung du lieu nang, khong commit vao repo.
SOURCES_DIR = Path(config.DATA_ROOT) / "kb_sources" / "raw"


# ----------------------------------------------------------------------------------------------
# Doc file cau hinh
# ----------------------------------------------------------------------------------------------
def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def load_sources():
    """{id: muc trong sources.yaml} (doc mot lan; khong sua dict tra ve)."""
    return {s["id"]: s for s in load_yaml(SOURCES_YAML)}


def load_lock():
    return json.loads(LOCK_JSON.read_text(encoding="utf-8")) if LOCK_JSON.exists() else {}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lf(path):
    """sha256 sau khi doi CRLF -> LF, cho nguon `local` (file trong repo): lock giong nhau tren Windows va Linux."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def is_local(src_id, name):
    return any(item["name"] == name for item in load_sources()[src_id].get("local", []))


def sha256_source(src_id, name):
    """Hash dung trong lock: file tai ve / dat tay -> sha256 nguyen byte; file trong repo -> sha256_lf."""
    p = source_path(src_id, name)
    return sha256_lf(p) if is_local(src_id, name) else sha256_file(p)


def source_path(src_id, name):
    """Duong dan file `name` cua nguon `src_id` (raw/<id>/<name>, hoac file trong repo voi nguon local)."""
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
CELL_SEP = " ¦ "      # ranh gioi o bang HTML: trich dan khong duoc vuot qua dau nay


def normalize(text):
    """Chuan hoa de so nguyen van (ap dung NHU NHAU cho trich dan va nguon):
    NFKC; bo '(Citation: ...)', link markdown '[x](url)' -> 'x' va the <code> cua ATT&CK; gop khoang trang;
    bo khoang trang truoc dau cau (sinh ra khi go the HTML)."""
    text = unicodedata.normalize("NFKC", text)
    text = _CITATION.sub("", text)
    text = _MDLINK.sub(r"\1", text)
    text = re.sub(r"</?code>", "", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r" ([,.;:)])", r"\1", text).strip()


def _html_text(raw):
    raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"</t[dh]\s*>", CELL_SEP, raw, flags=re.I)
    return html.unescape(_TAGS.sub(" ", raw))


def _stix_text(path):
    """Moi truong name + description cua bundle STIX (technique, analytic, mitigation, quan he), noi lai."""
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    return "\n".join(f"{o.get('name', '')}\n{o.get('description', '')}" for o in bundle["objects"])


def _xml_text(path):
    import xml.etree.ElementTree as ET
    return " ".join(ET.parse(path).getroot().itertext())


def _pdf_text(path):
    from pypdf import PdfReader
    return " ".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _file_text(p):
    if not p.exists():
        raise FileNotFoundError(f"thieu file nguon {p}; chay build/fetch_sources.py (hoac dat tay file `manual`)")
    if p.suffix == ".json":
        return _stix_text(p)
    if p.suffix == ".xml":
        return _xml_text(p)
    if p.suffix == ".html":
        return _html_text(p.read_text(encoding="utf-8", errors="replace"))
    if p.suffix == ".pdf":
        return _pdf_text(p)
    return p.read_text(encoding="utf-8", errors="replace")


@lru_cache(maxsize=None)
def source_text(src_id):
    """Toan bo van ban (da chuan hoa) cua mot nguon, dung de kiem trich dan."""
    return normalize("\n".join(_file_text(p) for p in source_files(src_id)))


@lru_cache(maxsize=None)
def file_text(src_id, name):
    """Van ban (da chuan hoa) cua MOT file trong nguon."""
    return normalize(_file_text(source_path(src_id, name)))


def locator_file(src_id, locator):
    """Ten file cua nguon ma locator chi toi ('BasicFlow.java, firstPacket()' -> 'BasicFlow.java'), hoac None."""
    if not locator:
        return None
    head = re.split(r"[,(]", locator, maxsplit=1)[0].strip()
    spec = load_sources()[src_id]
    names = [f["name"] for f in spec.get("files", []) + spec.get("local", [])]
    return head if head in names else None


def quote_ok(src_id, quote, locator=None):
    """Trich dan co nam nguyen van trong nguon khong (sau khi chuan hoa).
    Neu locator bat dau bang ten mot file cua nguon, chi kiem trong file do."""
    name = locator_file(src_id, locator)
    hay = file_text(src_id, name) if name else source_text(src_id)
    return normalize(quote) in hay


# ----------------------------------------------------------------------------------------------
# Mot doan van (passage) cua KB
# ----------------------------------------------------------------------------------------------
# M  khoa nhan -> technique (Borisenko 2022)      B  tri thuc theo technique (ATT&CK, CAPEC)
# H1 dinh nghia feature (+ co cua cot co)         H2 giao thuc, cong (IANA), co TCP (RFC)
# H3 artefact co dieu kien tren gia tri flow      G  mitigation ATT&CK (code dien, khong gui VLM)
PART_ORDER = ["M", "B", "H1", "H2", "H3", "G"]


def passage(pid, part, keys, kind, text, *, title=None, source=None, support=None, notes=None,
            when=None, for_vlm=True):
    """Mot doan cua store/passages.jsonl.

    kind:
      verbatim  `text` la chuoi con nguyen van cua nguon `source` ({src, locator}); validate.py kiem.
      rendered  `text` do CODE ghep tu cac truong nguyen van trong `support` (vd dong IANA, dong Table II).
      defined   `text` la dinh nghia viet trong definitions/*.yaml (feature, co, feature AE, artefact H3), kem `support`.
    notes: [{id, text, support, relation}] ghi chu gan them (H1).
    for_vlm: mac dinh True (moi phan, ke ca G: ung vien mitigation de VLM chon).
    """
    assert kind in ("verbatim", "rendered", "defined"), kind
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
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
