"""
G' — Ung pho: mitigation ATT&CK cua technique, theo quan he "mitigates" trong bundle v20.1.

Y tuong: muc 6 cua report do CODE dien (khong dua cho VLM, khong cham diem VLM). Lay TAT CA mitigation
lien ket voi technique (khong chon tay), moi muc: ten mitigation + mo ta cua quan he (nguyen van,
cat <= 80 tu).

    pid G-<technique>-<M-id>    khoa {"technique": T..., "mitigation": M...}    for_vlm = False
"""
from __future__ import annotations

import common as C
import mitre


def build(ctx):
    out = []
    for tid in sorted({m["technique"]["decision"] for m in ctx["mapping"].values()}):
        for mid, name, desc in mitre.mitigations(tid):
            quote = mitre.clip_sentences(desc, max_words=80)
            out.append(C.passage(f"G-{tid}-{mid}", "G", {"technique": tid, "mitigation": mid}, "rendered",
                                 f"{mid} {name}: {quote}", title=f"{mid} {name} (for {tid})",
                                 support=[{"src": "attack-20.1", "locator": f"relationship {mid} mitigates {tid}",
                                           "quote": quote}],
                                 for_vlm=False))
    return out
