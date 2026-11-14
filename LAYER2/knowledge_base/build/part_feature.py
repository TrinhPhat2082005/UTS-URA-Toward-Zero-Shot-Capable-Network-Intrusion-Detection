"""
Phan 2 cua KB (H1): dinh nghia feature, de VLM hieu moi chieu cua vector SHAP.

Dinh nghia theo dung phien ban CICFlowMeter-V3 ma CSE-CIC-IDS2018 dung (ma nguon @50d668db); khi ReadMe va ma
nguon cua phien ban do khac nhau thi theo ma nguon (phien ban dataset dung).
  - Cot co dinh nghia ReadMe va khong co ghi chu `replaces`: "<cot>: <dinh nghia ReadMe nguyen van>." (rendered)
  - Cot co ghi chu `replaces` (definitions/feature_notes.yaml): ReadMe khong dung voi phien ban nay ->
    text = "<cot>: <cac ghi chu replaces>" (defined, kem trich dan ma nguon / tai lieu doc lap)
  - Ghi chu `adds` (don vi, payload, goi dau cong hai lan, ...) noi sau dinh nghia.
  - Cot co (8 cot Flag Cnt + Fwd PSH/URG Flags): {holds}/{named} dien tu flag_order.flag_column_holds()
    (co ma cot THUC SU chua theo ma nguon 2018).
  - Dst Port, Protocol, ae_recon_error, ae_latent (16 chieu gop mot doan): definitions/model_features.yaml.
  - H1-DIRECTION: cau ReadMe dinh nghia chieu forward/backward (nguyen van).

    pid H1-<cot>   khoa {"feature": <cot>}   (doan latent: khoa {"feature": "ae_latent"})
"""
from __future__ import annotations

import common as C
from flag_order import FLAG_OF_NAME, flag_column_holds

DIRECTION_QUOTE = ("It can be used to generate bidirectional flows, where the first packet determines the forward "
                   "(source to destination) and backward (destination to source) directions, hence the statistical "
                   "time-related features can be calculated separately in the forward and backward directions.")
RELATIONS = ("replaces", "adds")


def _notes_for(col, ctx, holds):
    """Ghi chu cua cot (definitions/feature_notes.yaml). relation:
      replaces  dinh nghia ReadMe khong dung voi phien ban 2018 -> ghi chu thanh dinh nghia, ReadMe bi bo
      adds      dinh nghia ReadMe dung nhung thieu -> giu ReadMe, them ghi chu"""
    notes = []
    for n in ctx["feature_notes"]:
        if col not in n["applies_to"]:
            continue
        if n.get("relation") not in RELATIONS:
            raise ValueError(f"{n['id']}: relation phai thuoc {RELATIONS}")
        text = n["text"]
        if "{holds}" in text or "{named}" in text:
            held = holds.get(col)
            if n["id"] == "XN-FLAG-ORDER" and held == FLAG_OF_NAME.get(col):
                continue                                   # cot khong bi lech (ACK): khong can ghi chu thu tu
            text = text.format(holds=held, named=FLAG_OF_NAME.get(col, held))
        notes.append({"id": n["id"], "text": text, "support": n["support"], "relation": n["relation"]})
    return notes


def build(ctx):
    out, holds = [], flag_column_holds()
    model = {f: e for e in ctx["model_features"] for f in e["applies_to"]}
    csv_cols = {f["csv"] for f in ctx["feature_names"]}
    for f in ctx["feature_names"]:
        col = f["csv"]
        notes = _notes_for(col, ctx, holds)
        replaces = [n for n in notes if n["relation"] == "replaces"]
        if f["readme"] and replaces:
            text = f"{col}: " + " ".join(n["text"] for n in replaces)
            support = [s for n in replaces for s in n["support"]] + [f["code_2018"]]
            notes = [n for n in notes if n["relation"] != "replaces"]
            kind = "defined"
        elif f["readme"]:
            d = f["definition"].rstrip()
            text = f"{col}: {d}" + ("" if d.endswith((".", ")")) else ".")   # ReadMe khong co dau cham cuoi
            support = [f["readme"], f["code_2018"]]
            kind = "rendered"
        else:
            e = model[col]
            text, support, kind = e["text"], e["support"] + [f["code_2018"]], "defined"
        out.append(C.passage(f"H1-{C.pid_safe(col)}", "H1", {"feature": col, "group": f["group"]}, kind, text,
                             title=col, support=support, notes=notes))
    for e in ctx["model_features"]:
        if e["applies_to"][0] in csv_cols:
            continue                                   # Dst Port / Protocol da lam o tren
        key = "ae_latent" if e["pid"] == "H1-ae_latent" else e["applies_to"][0]
        out.append(C.passage(e["pid"], "H1", {"feature": key, "group": "model"}, "defined", e["text"],
                             title=key, support=e["support"]))
    out.append(C.passage("H1-DIRECTION", "H1", {"feature": "_direction", "group": "direction"}, "verbatim",
                         DIRECTION_QUOTE, title="Forward and backward direction",
                         source={"src": "cicflowmeter-readme", "locator": "ReadMe.txt, introduction"}))
    return out
