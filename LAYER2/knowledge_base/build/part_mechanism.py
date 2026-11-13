"""
B' — Co che: mo ta technique ATT&CK va attack pattern CAPEC, NGUYEN VAN.

Y tuong: muc 4 cua report doi chieu bang chung cua alert voi hanh vi ma technique/CAPEC mo ta.
  - B-<technique>: doan dau phan description cua technique trong ATT&CK v20.1 (bo "(Citation: ...)").
  - B-<CAPEC-id>:  phan Description cua muc CAPEC v3.9 da chon trong mapping.yaml, cat o ranh gioi cau
                   (<= 120 tu). Route "taxonomy_mapping" duoc kiem: CAPEC phai tu lien ket toi technique;
                   route "child_of_linked": CAPEC la con truc tiep cua mot muc lien ket toi technique.

    khoa {"technique": T...} hoac {"capec": CAPEC-...}
"""
from __future__ import annotations

import common as C
import mitre


def build(ctx):
    out, techs, capecs = [], {}, {}
    for label, m in ctx["mapping"].items():
        techs.setdefault(m["technique"]["decision"], []).append(label)
        cap = m["capec"]
        if cap["decision"]:
            tid = m["technique"]["decision"]
            if cap["route"] == "taxonomy_mapping" and not mitre.capec_maps_to(cap["decision"], tid):
                raise ValueError(f"{label}: {cap['decision']} khong lien ket toi {tid} trong CAPEC 3.9")
            # route child_of_linked (checklist 6.3): CAPEC khong tu lien ket, nhung la con truc tiep cua mot
            # muc lien ket toi technique (vd CAPEC-16 ChildOf CAPEC-49, CAPEC-49 -> T1110.001)
            if cap["route"] == "child_of_linked" and not mitre.capec_child_of_linked(cap["decision"], tid):
                raise ValueError(f"{label}: {cap['decision']} khong phai con cua muc CAPEC nao lien ket toi {tid}")
            capecs.setdefault(cap["decision"], {"labels": [], "route": cap["route"], "tech": m["technique"]["decision"]})
            capecs[cap["decision"]]["labels"].append(label)

    for tid in sorted(techs):
        t = mitre.technique(tid)
        text = mitre.first_paragraph(t["description"])
        out.append(C.passage(f"B-{tid}", "B", {"technique": tid}, "verbatim", text,
                             title=f"ATT&CK {tid} {t['name']}",
                             source={"src": "attack-20.1", "locator": f"{tid} description, first paragraph"}))
    for cid in sorted(capecs, key=lambda c: int(c.split("-")[1])):
        c = mitre.capec(cid)
        out.append(C.passage(f"B-{cid}", "B", {"capec": cid, "technique": capecs[cid]["tech"]}, "verbatim",
                             mitre.clip_sentences(c["description"]),
                             title=f"{cid} {c['name']}",
                             source={"src": "capec-3.9", "locator": f"{cid} Description"}))
    return out
