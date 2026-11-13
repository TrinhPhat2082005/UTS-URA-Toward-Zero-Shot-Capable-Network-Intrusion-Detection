"""
Buoc 2 cua KB: moi con so ve dataset dung trong KB deu tinh o day, tu train_full.csv (Protocol B).

    python LAYER2/knowledge_base/build/data_facts.py      # -> store/data_facts.json

Sua 2026-11-13 (checklist D1): so o day chi tinh tren train nen KHONG duoc dung lam bang chung trong doan gui
VLM (validate.py chan). Doan gui VLM chi dung claim co kiem tinh dung (build/claims.py -> store/claims.json).
File nay con dung de: (1) tim danh sach cot hang / cot trung nhau (claims.py kiem lai tren moi nguon),
(2) flag_column_holds (thu tu HashMap tinh lai tu ma nguon), (3) bang mo ta trong build_report.md.
Ty le lam tron 4 chu so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

R = 4
FLAG_CNT = ["FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
            "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt"]
DIR_FLAGS = ["Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags"]


def share(mask):
    return round(float(np.mean(mask)), R) if len(mask) else None


def java_hashmap_order(keys, capacity=16):
    """Thu tu duyet cua java.util.HashMap (OpenJDK 8+) voi cac khoa chuoi, chen theo thu tu `keys`.

    hash = h ^ (h >>> 16), bucket = hash & (capacity - 1); duyet bucket tang dan, trong bucket theo
    thu tu chen. 8 khoa < nguong resize 12 nen capacity giu 16.
    """
    def jhash(s):
        h = 0
        for ch in s:
            h = (31 * h + ord(ch)) & 0xFFFFFFFF
        return h
    buckets = {}
    for k in keys:
        h = jhash(k)
        buckets.setdefault((h ^ (h >> 16)) & (capacity - 1), []).append(k)
    return [k for b in sorted(buckets) for k in buckets[b]]


def main():
    df = pd.read_csv(C.config.TRAIN_FULL_CSV)
    lab = df["Label_Text"]
    feats = [c for c in df.columns if c not in ("Label", "Label_Text")]
    attack = lab != "Benign"
    tcp, udp = df["Protocol"] == 6, df["Protocol"] == 17
    zero_payload = (df["TotLen Fwd Pkts"] == 0) & (df["TotLen Bwd Pkts"] == 0)
    fwd_m1, bwd_m1 = df["Init Fwd Win Byts"] == -1, df["Init Bwd Win Byts"] == -1
    timeout = df["Flow Duration"] >= 119_000_000

    facts = {"dataset": {
        "file": "train_full.csv (Protocol B training split)",
        "sha256": C.sha256_file(C.config.TRAIN_FULL_CSV),
        "rows": int(len(df)), "feature_columns": len(feats),
        "label_counts": {k: int(v) for k, v in lab.value_counts().sort_index().items()},
    }}

    # --- cot hang va cot trung nhau ------------------------------------------------------------
    facts["constant_columns"] = {c: float(df[c].iloc[0]) for c in feats if df[c].nunique() == 1}
    nonconst = [c for c in feats if c not in facts["constant_columns"]]
    ident, near = [], []
    vals = {c: df[c].to_numpy() for c in nonconst}
    for i, a in enumerate(nonconst):
        for b in nonconst[i + 1:]:
            eq = float(np.mean(vals[a] == vals[b]))
            if eq == 1.0:
                ident.append([a, b])
            elif eq >= 0.999:                       # vd tong bi in dang so mu lam tron o vai dong
                near.append([a, b, round(eq, 6)])
    facts["identical_columns"] = ident
    facts["near_identical_columns"] = near
    facts["identical_with"], facts["near_identical_with"] = {}, {}
    for a, b in ident:
        facts["identical_with"].setdefault(a, []).append(b)
        facts["identical_with"].setdefault(b, []).append(a)
    for a, b, _ in near:
        facts["near_identical_with"].setdefault(a, []).append(b)
        facts["near_identical_with"].setdefault(b, []).append(a)

    # --- cot co (flag) -------------------------------------------------------------------------
    facts["flag_values"] = {c: sorted(float(v) for v in df[c].unique()) for c in FLAG_CNT + DIR_FLAGS}
    holds = java_hashmap_order(["FIN", "SYN", "RST", "PSH", "ACK", "URG", "CWR", "ECE"])
    header = dict(zip(FLAG_CNT, holds))     # cot CSV thu i nhan gia tri cua khoa thu i khi duyet HashMap
    facts["flag_column_holds"] = header
    facts["flag_consistency"] = {
        "SYN_Flag_Cnt_equals_Fwd_PSH_Flags": share(df["SYN Flag Cnt"] == df["Fwd PSH Flags"]),
        "CWE_Flag_Count_equals_Fwd_URG_Flags": share(df["CWE Flag Count"] == df["Fwd URG Flags"]),
        "RST_Flag_Cnt_equals_ECE_Flag_Cnt": share(df["RST Flag Cnt"] == df["ECE Flag Cnt"]),
    }
    facts["flag_share_tcp"] = {c: share(df.loc[tcp, c] > 0) for c in FLAG_CNT}

    # --- Init Win = -1, payload, thoi luong, protocol 0 ----------------------------------------
    facts["init_win_minus1"] = {
        "fwd_rows": int(fwd_m1.sum()),
        "fwd_udp_share": share(udp[fwd_m1]),
        "bwd_rows": int(bwd_m1.sum()),
        "bwd_zero_bwd_pkts_share": share(df.loc[bwd_m1, "Tot Bwd Pkts"] == 0),
        "bwd_udp_share": share(udp[bwd_m1]),
        "bwd_zero_bwd_pkts_or_udp_share": share((df.loc[bwd_m1, "Tot Bwd Pkts"] == 0) | udp[bwd_m1]),
    }
    act0 = (df["TotLen Fwd Pkts"] > 0) & (df["Fwd Act Data Pkts"] == 0)
    facts["fwd_payload_without_act_data"] = {
        "rows": int(act0.sum()),
        "single_fwd_pkt_share": share(df.loc[act0, "Tot Fwd Pkts"] == 1),
    }
    facts["flow_duration"] = {"max": int(df["Flow Duration"].max()),
                              "negative_rows": int((df["Flow Duration"] < 0).sum())}
    p0 = df["Protocol"] == 0
    facts["protocol_0"] = {"rows": int(p0.sum()), "dst_port_0_share": share(df.loc[p0, "Dst Port"] == 0),
                           "zero_payload_share": share(zero_payload[p0])}

    # --- cong chinh: chiem >= 1% flow cua it nhat mot nhan tan cong ----------------------------
    # (Infiltration quet hang nghin cong, nen khong dung nguong "phu 99% flow tan cong".)
    main_ports = {}
    for L in sorted(lab[attack].unique()):
        vc = df.loc[lab == L, "Dst Port"].value_counts(normalize=True)
        for p, v in vc[vc >= 0.01].items():
            main_ports.setdefault(str(int(p)), {})[L] = round(float(v), R)
    covered = df.loc[attack, "Dst Port"].astype(int).astype(str).isin(main_ports)
    facts["attack_main_ports"] = {"rule": "port carries >= 1% of the flows of at least one attack label",
                                  "ports": dict(sorted(main_ports.items(), key=lambda kv: int(kv[0]))),
                                  "attack_flow_coverage": share(covered)}
    ports = list(main_ports)

    # --- theo nhan ----------------------------------------------------------------------------
    by = {}
    for L in sorted(lab.unique()):
        m = lab == L
        pv = df.loc[m, "Dst Port"].value_counts(normalize=True).head(5)
        prot = df.loc[m, "Protocol"].value_counts(normalize=True)
        by[L] = {
            "n": int(m.sum()),
            "top_ports": {str(int(p)): round(float(v), R) for p, v in pv.items()},
            "protocol": {str(int(p)): round(float(v), R) for p, v in prot.items()},
            "zero_payload": share(zero_payload[m]),
            "init_fwd_win_minus1": share(fwd_m1[m]),
            "init_bwd_win_minus1": share(bwd_m1[m]),
            "duration_ge_119s": share(timeout[m]),
            "udp": share(udp[m]),
        }
    facts["by_label"] = by

    C.write_json(C.FACTS_JSON, facts)
    print(f"-> {C.FACTS_JSON}")
    print(f"   cot hang: {list(facts['constant_columns'])}")
    print(f"   cot trung: {ident}")
    print(f"   flag_consistency: {facts['flag_consistency']}")
    print(f"   flag_share_tcp: {facts['flag_share_tcp']}")
    print(f"   init_win_minus1: {facts['init_win_minus1']}")
    print(f"   fwd_payload_without_act_data: {facts['fwd_payload_without_act_data']}")
    print(f"   protocol_0: {facts['protocol_0']}")
    print(f"   attack_main_ports: {ports} (phu {facts['attack_main_ports']['attack_flow_coverage']})")


if __name__ == "__main__":
    main()
