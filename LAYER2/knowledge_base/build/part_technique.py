"""
Phan 1 cua KB: tri thuc theo technique ATT&CK. Khoa DUY NHAT: nhan -> technique theo Table II cua Borisenko et al.
2022 (mapping/borisenko2022_table2.yaml). Moi noi dung sau khoa lay tu MITRE theo technique ID, khong dung gi
cua dataset:

  M-<nhan>         rendered   "Reference label X is mapped by Borisenko et al. (2022), Table II, to ..." (ten tactic/
                              technique lay tu ATT&CK v20.1 theo ID; ID lay tu Table II)
  B-<tid>          verbatim   toan bo description cua technique trong ATT&CK v20.1 (bo "(Citation: ...)")
  B-DET-<tid>      rendered   cac analytic cua technique co nguon log la data component mang (Network Traffic Flow,
                              Network Connection Creation, Network Traffic Content), moi mo ta nguyen van
  B-<CAPEC-id>     verbatim   description cua moi muc CAPEC v3.9 ma CAPEC tu lien ket toi technique
                              (Taxonomy_Mapping ATTACK), cat o ranh gioi cau <= 120 tu

Technique cua mot entry = sub_technique neu co, nguoc lai technique. Khong co lua chon cua tac gia: moi analytic
mang va moi CAPEC lien ket deu duoc lay.
"""
from __future__ import annotations

import common as C
import mitre


def technique_ids(table):
    """{nhan: [technique ID KB dung]} theo thu tu cac entry cua Table II."""
    return {lab: [e["sub_technique"] or e["technique"] for e in row["entries"]] for lab, row in table["labels"].items()}


def _mapping_text(label, row):
    parts = []
    for e in row["entries"]:
        s = f"tactic {mitre.tactic_name(e['tactic'])} ({e['tactic']}), technique {mitre.technique(e['technique'])['name']} ({e['technique']})"
        if e["sub_technique"]:
            s += f", sub-technique {mitre.technique(e['sub_technique'])['name']} ({e['sub_technique']})"
        parts.append(s)
    return f"Reference label {label} is mapped by Borisenko et al. (2022), Table II, to " + "; and to ".join(parts) + "."


def build(ctx):
    table = ctx["mapping"]
    out = []
    # --- M: khoa nhan -> technique ------------------------------------------------------------------
    for label, row in table["labels"].items():
        out.append(C.passage(f"M-{label}", "M", {"label": label}, "rendered", _mapping_text(label, row),
                             title=f"ATT&CK mapping of {label}", support=[row["row"]]))
    # --- B: theo technique --------------------------------------------------------------------------
    tids = sorted({t for ts in technique_ids(table).values() for t in ts})
    for tid in tids:
        t = mitre.technique(tid)
        out.append(C.passage(f"B-{tid}", "B", {"technique": tid}, "verbatim", t["description"],
                             title=f"ATT&CK {tid} {t['name']}",
                             source={"src": "attack-20.1", "locator": f"{tid} description"}))
        an = mitre.network_analytics(tid)
        if an:
            # ID / ten technique chi nam o title (retrieve.py an title khi bat co zero-day)
            text = ("ATT&CK detection analytics of this technique that use network data sources: "
                    + " ".join(f"{a['id']} ({', '.join(a['components'])}): {C.normalize(a['description'])}" for a in an))
            out.append(C.passage(f"B-DET-{tid}", "B", {"technique": tid, "detection": True}, "rendered", text,
                                 title=f"ATT&CK detection analytics for {tid}",
                                 support=[{"src": "attack-20.1", "locator": f"analytic {a['id']} (detection strategy of {tid})",
                                           "quote": a["description"]} for a in an]))
        for c in mitre.capec_linked(tid):
            out.append(C.passage(f"B-{c['id']}", "B", {"capec": c["id"], "technique": tid}, "verbatim",
                                 mitre.clip_sentences(c["description"]), title=f"{c['id']} {c['name']}",
                                 source={"src": "capec-3.9", "locator": f"{c['id']} Description (CAPEC maps it to {tid})"}))
    # mot muc CAPEC co the lien ket toi nhieu technique: giu mot doan
    seen, uniq = set(), []
    for p in out:
        if p["pid"] not in seen:
            seen.add(p["pid"])
            uniq.append(p)
    return uniq
