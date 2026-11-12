"""
H1 — Dinh nghia feature: moi chieu cua vector SHAP co mot doan.

Y tuong: muc 3 cua report giai thich vi sao gate canh bao bang cac feature co |phi| lon nhat; VLM can
biet feature do la gi, don vi gi, va cong cu trich xuat tinh no the nao trong dataset nay.

  - 78 cot CSV (authored/feature_names.yaml, sinh bang may):  "<cot>: <dinh nghia ReadMe nguyen van>"
    (rendered); 2 cot khong co trong ReadMe (Dst Port, Protocol) lay tu authored/extra_features.yaml.
  - Ghi chu (notes) gan vao doan:
      * authored/extractor_notes.yaml  (cach CICFlowMeter 2018 tinh cot, co trich ma nguon)
      * sinh tu data_facts: cot trung / gan trung voi cot khac trong dataset
  - Chieu cua Layer 1 (ae_recon_error; 16 ae_latent gop thanh 1 doan): authored/extra_features.yaml.
  - H1-DIRECTION: cau ReadMe dinh nghia chieu forward/backward (nguyen van), chon khi top feature co Fwd/Bwd.

    pid H1-<cot>   khoa {"feature": <cot>}   (doan latent: khoa {"feature": "ae_latent"})
"""
from __future__ import annotations

import common as C

FLAG_OF_COLUMN = {"FIN Flag Cnt": "FIN", "SYN Flag Cnt": "SYN", "RST Flag Cnt": "RST", "PSH Flag Cnt": "PSH",
                  "ACK Flag Cnt": "ACK", "URG Flag Cnt": "URG", "CWE Flag Count": "CWR", "ECE Flag Cnt": "ECE"}

DIRECTION_QUOTE = ("It can be used to generate bidirectional flows, where the first packet determines the forward "
                   "(source to destination) and backward (destination to source) directions, hence the statistical "
                   "time-related features can be calculated separately in the forward and backward directions.")


def _notes_for(col, ctx):
    facts, notes = ctx["facts"], []
    for n in ctx["extractor_notes"]:
        if col not in n["applies_to"]:
            continue
        text = n["text"]
        if "{holds}" in text:
            holds = facts["flag_column_holds"][col]
            if holds == FLAG_OF_COLUMN[col]:
                continue
            text = text.format(named=f"'{col}'", holds=holds)
        notes.append({"id": n["id"], "text": text, "support": n["support"]})
    for other in facts["identical_with"].get(col, []):
        notes.append({"id": "DATA-IDENTICAL", "text": f"In this dataset this column is identical to '{other}' in every row.",
                      "support": [{"data": f"identical_with.{col}"}]})
    for other in facts["near_identical_with"].get(col, []):
        notes.append({"id": "DATA-NEAR-IDENTICAL",
                      "text": f"In this dataset this column equals '{other}' in at least 99.9% of rows.",
                      "support": [{"data": f"near_identical_with.{col}"}]})
    return notes


def build(ctx):
    out = []
    extra = {f: e for e in ctx["extra_features"] for f in e["applies_to"]}
    for f in ctx["feature_names"]:
        col = f["csv"]
        if f["readme"]:
            d = f["definition"].rstrip()
            text = f"{col}: {d}" + ("" if d.endswith((".", ")")) else ".")   # ReadMe khong co dau cham cuoi
            support = [f["readme"], f["code_2018"]] + ([f["code_master"]] if f.get("code_master") else [])
            kind = "rendered"
        else:
            e = extra[col]
            text, support, kind = e["text"], e["support"] + [f["code_2018"]], "authored"
        out.append(C.passage(f"H1-{C.pid_safe(col)}", "H1", {"feature": col, "group": f["group"]}, kind, text,
                             title=col, support=support, notes=_notes_for(col, ctx)))
    for e in ctx["extra_features"]:
        if e["applies_to"][0] in {f["csv"] for f in ctx["feature_names"]}:
            continue                                   # Dst Port / Protocol da lam o tren
        key = "ae_latent" if e["pid"] == "H1-ae_latent" else e["applies_to"][0]
        out.append(C.passage(e["pid"], "H1", {"feature": key, "group": "model"}, "authored", e["text"],
                             title=key, support=e["support"]))
    out.append(C.passage("H1-DIRECTION", "H1", {"feature": "_direction", "group": "direction"}, "verbatim",
                         DIRECTION_QUOTE, title="Forward and backward direction",
                         source={"src": "cicflowmeter-readme", "locator": "ReadMe.txt, introduction"}))
    return out
