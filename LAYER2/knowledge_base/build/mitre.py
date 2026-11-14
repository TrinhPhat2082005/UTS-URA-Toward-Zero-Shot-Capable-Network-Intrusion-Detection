"""
Doc ATT&CK v20.1 (bundle STIX 2.1, JSON) va CAPEC v3.9 (XML) bang thu vien chuan.

Khong dung mitreattack-python: ca phan doc chi la tim object theo external_id va theo quan he
"mitigates", du ngan de nguoi duyet doc het.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from functools import lru_cache

import common as C

CAPEC_NS = {"c": "http://capec.mitre.org/capec-3"}


# ----------------------------------------------------------------------------------------------
# ATT&CK
# ----------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _attack():
    path = C.source_path("attack-20.1", "enterprise-attack-20.1.json")
    objs = json.loads(path.read_text(encoding="utf-8"))["objects"]
    live = [o for o in objs if not o.get("revoked") and not o.get("x_mitre_deprecated")]
    return live


def _ext_id(obj):
    return next((r.get("external_id") for r in obj.get("external_references", [])
                 if r.get("source_name") == "mitre-attack"), None)


@lru_cache(maxsize=None)
def technique(tid):
    """{'id','name','tactics','description','stix_id'} cua technique (vd 'T1499.002')."""
    for o in _attack():
        if o["type"] == "attack-pattern" and _ext_id(o) == tid:
            return {"id": tid, "name": o["name"], "stix_id": o["id"],
                    "tactics": [p["phase_name"] for p in o.get("kill_chain_phases", [])],
                    "description": o.get("description", "")}
    raise KeyError(f"ATT&CK 20.1 khong co technique {tid}")


def first_paragraph(text):
    return text.strip().split("\n\n")[0].strip()


@lru_cache(maxsize=None)
def tactic_name(ta_id):
    """Ten tactic ATT&CK theo ID (vd 'TA0040' -> 'Impact')."""
    for o in _attack():
        if o["type"] == "x-mitre-tactic" and _ext_id(o) == ta_id:
            return o["name"]
    raise KeyError(f"ATT&CK 20.1 khong co tactic {ta_id}")


NETWORK_COMPONENT_PREFIX = "Network"        # DC0078 Network Traffic Flow, DC0082 Network Connection Creation, DC0085 ...


def network_analytics(tid):
    """Analytic cua technique ma DA SO nguon log la data component mang (ten bat dau bang 'Network').

    He thong chi thay flow, khong thay tien trinh hay file tren may. Analytic chu yeu dua vao nguon log cua may
    (Process Creation, File Creation, Application Log Content...) mo ta buoc khong quan sat duoc tu flow, nen bi bo:
    chung chi lam dai context va de dan toi suy dien sai. Nguong: so component mang >= mot nua tong so component.

    Duong noi trong ATT&CK v18+: x-mitre-detection-strategy --detects--> technique; strategy.x_mitre_analytic_refs
    -> x-mitre-analytic; analytic.x_mitre_log_source_references[].x_mitre_data_component_ref -> data component.
    Tra [{id, platforms, components, description}] sap theo ID, bo mo ta trung lap (cung text, khac platform)."""
    t = technique(tid)
    by_id = {o["id"]: o for o in _attack()}
    comp = {o["id"]: o["name"] for o in _attack() if o["type"] == "x-mitre-data-component"}
    strategies = [by_id[r["source_ref"]] for r in _attack()
                  if r["type"] == "relationship" and r.get("relationship_type") == "detects"
                  and r["target_ref"] == t["stix_id"] and r["source_ref"] in by_id]
    out, seen = [], set()
    for s in strategies:
        for aid in s.get("x_mitre_analytic_refs", []):
            a = by_id.get(aid)
            if a is None:
                continue
            comps = sorted({comp.get(l.get("x_mitre_data_component_ref"), "")
                            for l in a.get("x_mitre_log_source_references", [])} - {""})
            n_net = sum(c.startswith(NETWORK_COMPONENT_PREFIX) for c in comps)
            if not comps or n_net * 2 < len(comps):
                continue
            desc = a.get("description", "").strip()
            if not desc or desc in seen:
                continue
            seen.add(desc)
            out.append({"id": _ext_id(a), "platforms": a.get("x_mitre_platforms", []), "components": comps,
                        "description": desc})
    return sorted(out, key=lambda x: x["id"] or "")


def mitigations(tid):
    """[(M-id, ten, mo ta quan he)] cua technique, theo quan he 'mitigates', sap theo M-id."""
    t = technique(tid)
    coa = {o["id"]: o for o in _attack() if o["type"] == "course-of-action"}
    out = []
    for r in _attack():
        if r["type"] == "relationship" and r.get("relationship_type") == "mitigates" and r["target_ref"] == t["stix_id"]:
            c = coa.get(r["source_ref"])
            if c is not None and _ext_id(c):
                out.append((_ext_id(c), c["name"], r.get("description", "")))
    return sorted(out)


# ----------------------------------------------------------------------------------------------
# CAPEC
# ----------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _capec():
    root = ET.parse(C.source_path("capec-3.9", "capec_v3.9.xml")).getroot()
    out = {}
    for ap in root.find("c:Attack_Patterns", CAPEC_NS):
        desc = ap.find("c:Description", CAPEC_NS)
        maps = []
        for m in ap.findall(".//c:Taxonomy_Mapping", CAPEC_NS):
            e = m.find("c:Entry_ID", CAPEC_NS)
            if m.get("Taxonomy_Name") == "ATTACK" and e is not None and e.text:
                maps.append("T" + e.text.strip())
        parents = [f"CAPEC-{r.get('CAPEC_ID')}" for r in ap.findall(".//c:Related_Attack_Pattern", CAPEC_NS)
                   if r.get("Nature") == "ChildOf"]
        # Mitigations: noi cac nut con bang " " giong common._xml_text, de text van la chuoi con nguyen van cua nguon
        mitig = [C.normalize(" ".join(m.itertext())) for m in ap.findall("c:Mitigations/c:Mitigation", CAPEC_NS)]
        out[f"CAPEC-{ap.get('ID')}"] = {
            "id": f"CAPEC-{ap.get('ID')}", "name": ap.get("Name"), "status": ap.get("Status"),
            "description": " ".join("".join(desc.itertext()).split()) if desc is not None else "",
            "attack": maps, "parents": parents, "mitigations": [m for m in mitig if m],
        }
    return out


def capec(cid):
    return _capec()[cid]


def capec_linked(tid):
    """Moi muc CAPEC v3.9 ma CAPEC tu lien ket toi technique `tid` (Taxonomy_Mapping ATTACK), theo so thu tu."""
    return sorted((c for c in _capec().values() if tid in c["attack"]), key=lambda c: int(c["id"].split("-")[1]))


# ----------------------------------------------------------------------------------------------
def clip_sentences(text, max_words=120):
    """Cat o ranh gioi cau, toi da max_words tu; ket qua van la chuoi con nguyen van cua text."""
    sents = re.split(r"(?<=[.!?])\s+", C.normalize(text))
    out, n = [], 0
    for s in sents:
        w = len(s.split())
        if out and n + w > max_words:
            break
        out.append(s)
        n += w
    return " ".join(out)
