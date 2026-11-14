"""
G — Bien phap giam thieu (ung vien), theo technique ID cua nhan (Borisenko 2022, Table II). Hai nguon MITRE, deu
di theo lien ket MITRE tu khai bao (khong co lua chon cua tac gia):

  G-<T>-<M-id>          ATT&CK v20.1: quan he "<M-id> mitigates <T>"; text = ten mitigation + mo ta cua quan he
                        (nguyen van, cat o ranh gioi cau <= 80 tu)                               kind rendered
  G-<T>-<CAPEC-n>-<k>   CAPEC v3.9: muc Mitigation thu k cua moi muc CAPEC ma CAPEC lien ket toi <T>
                        (Taxonomy_Mapping ATTACK); text nguyen van, cat <= 80 tu                 kind verbatim

Ca hai duoc dua cho VLM nhu danh sach UNG VIEN: muc Mitigations cua bao cao chon cac muc phu hop voi bang chung
cua alert va trich [K:G-...] (danh gia faithfulness / groundedness). Khi bat co zero-day khong co G.

    khoa {"technique": T, "mitigation": M-id | CAPEC-n#k, "source": "attack" | "capec"}
"""
from __future__ import annotations

import common as C
import mitre
from part_technique import technique_ids


def build(ctx):
    out = []
    for tid in sorted({t for ts in technique_ids(ctx["mapping"]).values() for t in ts}):
        # --- ATT&CK: quan he mitigates ----------------------------------------------------------
        for mid, name, desc in mitre.mitigations(tid):
            quote = mitre.clip_sentences(desc, max_words=80)
            out.append(C.passage(f"G-{tid}-{mid}", "G", {"technique": tid, "mitigation": mid, "source": "attack"},
                                 "rendered", f"{mid} {name}: {quote}", title=f"{mid} {name} (for {tid})",
                                 support=[{"src": "attack-20.1", "locator": f"relationship {mid} mitigates {tid}",
                                           "quote": quote}]))
        # --- CAPEC: truong Mitigations cua cac muc CAPEC lien ket toi technique ---------------------
        for c in mitre.capec_linked(tid):
            for k, m in enumerate(c["mitigations"], 1):
                out.append(C.passage(f"G-{tid}-{c['id']}-{k}", "G",
                                     {"technique": tid, "mitigation": f"{c['id']}#{k}", "source": "capec"},
                                     "verbatim", mitre.clip_sentences(m, max_words=80),
                                     title=f"{c['id']} {c['name']}, mitigation {k} (for {tid})",
                                     source={"src": "capec-3.9", "locator": f"{c['id']} Mitigations, item {k}"}))
    return out
