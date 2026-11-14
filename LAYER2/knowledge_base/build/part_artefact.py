"""
H3 — Artefact co dieu kien: gia tri la do cong cu trich xuat sinh ra.

Y tuong: phan canh bao cua report phai noi khi bang chung cua alert bi gioi han boi cach trich xuat
(vd flow khong co payload thi khong the thay cau lenh SQL). Doan chi duoc chon khi dieu kien `when` dung
voi gia tri tho cua alert (knowledge_base/conditions.py); `when: null` -> luon chon. Noi dung lay tu
definitions/artefacts.yaml (minh bach: dieu kien va co che ghi ro).

    pid H3-...   khoa {"artefact": pid}   when = dieu kien co cau truc hoac None
"""
from __future__ import annotations

import common as C
from knowledge_base import conditions


def build(ctx):
    cols = {f["csv"] for f in ctx["feature_names"]}
    out = []
    for a in ctx["artefacts"]:
        bad = [c for c in conditions.columns(a["when"]) if c not in cols]
        if bad:
            raise ValueError(f"{a['pid']}: dieu kien dung cot khong ton tai {bad}")
        out.append(C.passage(a["pid"], "H3", {"artefact": a["pid"]}, "defined", a["text"],
                             title=a["pid"], support=a["support"], when=a["when"]))
    return out
