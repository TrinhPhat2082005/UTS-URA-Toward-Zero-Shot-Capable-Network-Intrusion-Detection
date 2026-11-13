"""
A' — The nhan (label card): moi nhan mot doan.

Y tuong: VLM can biet nhan l* (do vote 5-NN tra ve) nghia la HANH VI gi va thuoc technique nao.
Doan nay ghep (rendered) tu authored/mapping.yaml: cau hanh vi cua tac gia + ten technique lay tu
ATT&CK 20.1. Khong nhac ten tool (ten tool chi nam trong trich dan scenario cua mapping.yaml).

Sua 2026-11-13:
  - checklist 1.3: the mo dau bang "Reference label <nhan>": l* la nhan cua cac tien le gan nhat trong R,
    khong phai nhan that cua flow.
  - checklist 1.6: cau technique KHONG nam trong `text` ma trong truong rieng `technique_text`;
    retrieve.py chi noi no vao khi alert KHONG bi bat co (khi bat co, report khong duoc gan technique).

    pid  A-<nhan>        khoa {"label": <nhan>}
"""
from __future__ import annotations

import common as C
import mitre


def build(ctx):
    out = []
    for label, m in ctx["mapping"].items():
        t = mitre.technique(m["technique"]["decision"])
        tactic = ", ".join(x.replace("-", " ") for x in t["tactics"])
        text = f"Reference label {label} (CSE-CIC-IDS2018): {m['behaviour']['text']}"
        technique_text = f"ATT&CK technique of this label: {t['id']} {t['name']} (tactic: {tactic})."
        support = (m["behaviour"]["support"]
                   + [{"decision": t["id"], "support": m["technique"]["support"],
                       "rationale": m["technique"]["rationale"]}])
        p = C.passage(f"A-{label}", "A", {"label": label}, "authored", text,
                      title=f"{label}: {t['name']}", support=support)
        p["technique_text"] = technique_text
        out.append(p)
    return out
