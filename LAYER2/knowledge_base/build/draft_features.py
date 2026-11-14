"""
Sinh definitions/feature_names.yaml BANG MAY cho 78 cot CSV (khong go tay ten hay dinh nghia nao).

Chuoi bang chung cho moi cot:
    header CSV  --(trung tung chu)-->  FlowFeature.java @2018 (cicflowmeter-src-2018)   vd tot_fw_pkt("Tot Fwd Pkts","TFwP")  //9
                --(cung so thu tu //9)-->  FlowFeature.java @master (cicflowmeter-readme) vd tot_fw_pkt("Total Fwd Packet","TFwP")
                --(trung ten, khong phan biet hoa/thuong)-->  dong trong ReadMe.txt   "total Fwd Packet  Total packets in the forward direction"

    python LAYER2/knowledge_base/build/draft_features.py            # ghi definitions/feature_names.yaml
    python LAYER2/knowledge_base/build/draft_features.py --check    # chi in, khong ghi

`group` va `discrete` la phan loai cua tac gia (dung de gom nhom H1 va de bang doi chieu C biet cot nao
khong lay trung binh duoc); chung khong phai thong tin dua cho VLM. Cot khong co trong ReadMe
(Dst Port, Protocol) de readme: null; dinh nghia cua chung lay tu trang UNB (xem part_feature.py).
"""
from __future__ import annotations

import argparse
import re

import pandas as pd
import yaml

import common as C

ENTRY = re.compile(r'^\s*(\w+)\("([^"]+)","([^"]+)"[^)]*\),?.*?//\s*(\d+)')

# Cot roi rac: gia tri la nhan/ma (cong, giao thuc, co 0/1) hoac co gia tri thay the -1 (Init Win),
# nen bang doi chieu C ghi tan suat gia tri thay vi trung binh.
DISCRETE = {"Dst Port", "Protocol", "Init Fwd Win Byts", "Init Bwd Win Byts"}


def group_of(col):
    c = col.lower()
    rules = [("port", c == "dst port"), ("protocol", c == "protocol"),
             ("bulk", "/b avg" in c or "blk rate" in c), ("subflow", c.startswith("subflow")),
             ("flag", "flag" in c), ("window", "init" in c and "win" in c),
             ("rate", c.endswith("/s")), ("time", "iat" in c or "duration" in c or c.startswith(("active", "idle"))),
             ("header", "header" in c), ("ratio", "ratio" in c),
             ("count", c in ("tot fwd pkts", "tot bwd pkts", "fwd act data pkts")),
             ("length", True)]
    return next(g for g, ok in rules if ok)


def parse_flowfeature(path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ENTRY.match(line)
        if m:
            ident, name, abbr, ordinal = m.groups()
            out[int(ordinal)] = {"id": ident, "name": name, "quote": f'{ident}("{name}","{abbr}"'}
    return out


def parse_readme(path):
    rows, on = {}, False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Feature Name"):
            on = True
            continue
        if on and line.startswith("-----"):
            break
        if on and line.strip():
            name, desc = re.split(r"\t+", line.strip(), maxsplit=1)
            rows[" ".join(name.split()).lower()] = {"name": name.strip(), "description": desc.strip(),
                                                   "quote": f"{name.strip()} {desc.strip()}"}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    cols = [c for c in pd.read_csv(C.config.TRAIN_FULL_CSV, nrows=0).columns if c not in ("Label", "Label_Text")]
    v2018 = parse_flowfeature(C.source_path("cicflowmeter-src-2018", "FlowFeature.java"))
    master = parse_flowfeature(C.source_path("cicflowmeter-readme", "FlowFeature.java"))
    readme = parse_readme(C.source_path("cicflowmeter-readme", "ReadMe.txt"))
    by_name_2018 = {v["name"]: (k, v) for k, v in v2018.items()}

    entries, problems = [], []
    for col in cols:
        if col not in by_name_2018:
            problems.append(f"{col}: khong co trong FlowFeature.java @2018")
            continue
        ordinal, e18 = by_name_2018[col]
        em = master.get(ordinal)
        rd = readme.get(" ".join(em["name"].split()).lower()) if em else None
        entry = {
            "csv": col,
            "group": group_of(col),
            "discrete": col in DISCRETE or group_of(col) == "flag",
            "code_2018": {"src": "cicflowmeter-src-2018", "locator": f"FlowFeature.java, feature //{ordinal}",
                          "quote": e18["quote"]},
            "code_master": ({"src": "cicflowmeter-readme", "locator": f"FlowFeature.java, feature //{ordinal}",
                             "quote": em["quote"]} if em else None),
            "readme": ({"src": "cicflowmeter-readme", "locator": "ReadMe.txt, list of extracted features",
                        "quote": rd["quote"]} if rd else None),
            "definition": rd["description"] if rd else None,
        }
        if em and em["id"] != e18["id"]:
            entry["note_rename"] = f"enum id doi ten giua hai ban: {e18['id']} -> {em['id']}"
        if rd is None:
            problems.append(f"{col}: khong co dong ReadMe (ten master: {em['name'] if em else None})")
        entries.append(entry)

    print(f"{len(entries)}/{len(cols)} cot ghep duoc voi ma nguon 2018")
    for p in problems:
        print("  -", p)
    if args.check:
        return
    header = ("# SINH BANG MAY boi build/draft_features.py; sua tay chi o 'group'/'discrete' (phan loai cua tac gia).\n"
              "# Moi quote duoc validate.py so nguyen van voi file nguon da khoa sha256.\n")
    path = C.DEFINITIONS_DIR / "feature_names.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=200),
                    encoding="utf-8")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
