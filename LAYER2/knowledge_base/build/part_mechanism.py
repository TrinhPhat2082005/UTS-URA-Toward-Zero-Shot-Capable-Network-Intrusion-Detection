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
            # route description (checklist 5.4): chi hop le khi CAPEC khong lien ket muc nao toi technique
            if cap["route"] == "description" and mitre.capec_linked(tid):
                raise ValueError(f"{label}: route 'description' nhung CAPEC co lien ket toi {tid}: "
                                 f"{[c['id'] for c in mitre.capec_linked(tid)]}")
            capecs.setdefault(cap["decision"], {"labels": [], "route": cap["route"], "tech": m["technique"]["decision"]})
            capecs[cap["decision"]]["labels"].append(label)
        # checklist 6.2: doan B' phu (vd T1018 cho Infiltration), khong doi technique cua nhan
        for ex in m.get("mechanism_extra", []):
            techs.setdefault(ex["decision"], []).append(label)

    mech = ctx.get("mechanism") or {}
    for tid in sorted(techs):
        t = mitre.technique(tid)
        # checklist 6.5: chon doan theo authored/mechanism.yaml (mac dinh doan dau); doan phai lien nhau
        paras = [p.strip() for p in t["description"].strip().split("\n\n")]
        sel = (mech.get(tid, {}).get("paragraphs") or {}).get("decision", [0])
        if sel != list(range(sel[0], sel[-1] + 1)) or sel[-1] >= len(paras):
            raise ValueError(f"mechanism.yaml {tid}: doan {sel} phai lien nhau va < {len(paras)}")
        text = " ".join(paras[i] for i in sel)
        loc = (f"{tid} description, first paragraph" if sel == [0]
               else f"{tid} description, paragraphs {sel[0] + 1}-{sel[-1] + 1}")
        out.append(C.passage(f"B-{tid}", "B", {"technique": tid}, "verbatim", text,
                             title=f"ATT&CK {tid} {t['name']}",
                             source={"src": "attack-20.1", "locator": loc}))
    for cid in sorted(capecs, key=lambda c: int(c.split("-")[1])):
        c = mitre.capec(cid)
        out.append(C.passage(f"B-{cid}", "B", {"capec": cid, "technique": capecs[cid]["tech"]}, "verbatim",
                             mitre.clip_sentences(c["description"]),
                             title=f"{cid} {c['name']}",
                             source={"src": "capec-3.9", "locator": f"{cid} Description"}))
    return out
