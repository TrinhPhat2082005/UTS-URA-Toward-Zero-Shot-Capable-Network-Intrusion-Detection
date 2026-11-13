"""
Buoc 2b cua KB (checklist D1): CLAIM so lieu co KIEM TINH DUNG.

    python LAYER2/knowledge_base/build/claims.py          # tinh lai train/val, giu gia tri raw da co
    python LAYER2/knowledge_base/build/claims.py --raw    # doc them 10 file CSV goc ($UTS_BENCH/01_raw_data/raw_csv)

Luat (da chot 2026-11-13): mot con so chi duoc vao doan gui VLM neu no DUNG tren train_full, val_full va
tung file CSV goc da cong bo (va hai nua cua moi nhan trong tung file, theo thu tu dong). Con so khong dat
thi bo, hoac doi thanh phat bieu bat bien. Doan gui VLM chi duoc tham chieu {data: claims.<id>} co
holds_all = true (validate.py kiem).

Moi claim:
    kind  invariant   moi dong thoa (gia tri = ty le dong thoa, phai = 1)
          threshold   ty le so voi nguong (vd >= 0.999, > 0.95)
          top         gia tri pho bien nhat bang mot gia tri cho truoc
          exists      ton tai o cap nhan: dung tren train, val va hop cac ngay (khong doi hoi tung ngay)
    scope "rows" (moi dong) hoac ten nhan (chi cac dong cua nhan do)

File goc: bo dong header lap va dong co NaN/Inf, bo dong trung trong ngay (78 feature + nhan), giong buoc
lam sach du lieu. 7/10 file goc dung o 1.048.575 dong (gioi han Excel) nen claim chi noi ve
"the published CSV files", khong noi ve toan bo luu luong cua ngay do.

    -> store/claims.json   {id: {statement, kind, scope, test, values: {nguon: gia tri}, holds_all, failing}}
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

CLAIMS_JSON = C.STORE_DIR / "claims.json"
MIN_N = 20                     # nhan co it hon MIN_N dong o mot nguon thi nguon do khong tinh cho nhan do
LABEL_COL_RAW = "Label"

FLAG_CNT = ["FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
            "CWE Flag Count", "ECE Flag Cnt"]
DIR_FLAGS = ["Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags"]
BULK = ["Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg"]
NEED = (["Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts",
         "TotLen Bwd Pkts", "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min",
         "Pkt Size Avg", "Pkt Len Mean", "Down/Up Ratio", "Subflow Fwd Pkts", "Subflow Bwd Pkts",
         "Subflow Fwd Byts", "Subflow Bwd Byts"] + FLAG_CNT + DIR_FLAGS + BULK)

RAW2STD = {"benign": "Benign", "ftpbruteforce": "FTP_BruteForce", "sshbruteforce": "SSH_Bruteforce",
           "dosattacksgoldeneye": "DoS_attacks_GoldenEye", "dosattacksslowloris": "DoS_attacks_Slowloris",
           "dosattacksslowhttptest": "DoS_attacks_SlowHTTPTest", "dosattackshulk": "DoS_attacks_Hulk",
           "ddosattacksloichttp": "DDoS_attacks_LOIC_HTTP", "ddosattackloicudp": "DDOS_attack_LOIC_UDP",
           "ddosattackhoic": "DDOS_attack_HOIC", "bruteforceweb": "Brute_Force_Web",
           "bruteforcexss": "Brute_Force_XSS", "sqlinjection": "SQL_Injection", "infilteration": "Infiltration",
           "infiltration": "Infiltration", "bot": "Bot"}


def std_label(s):
    return RAW2STD.get(re.sub(r"[^a-z]", "", str(s).lower()), str(s))


def _facts_lists():
    f = C.load_facts()
    pairs = [tuple(p) for p in f["identical_columns"]] + [tuple(p[:2]) for p in f["near_identical_columns"]]
    return pairs, f["constant_columns"], f["identical_columns"], f["near_identical_columns"]


PAIRS, CONSTANTS, IDENT, NEAR = _facts_lists()


# ----------------------------------------------------------------------------------------------
# Khung gon: moi nguon dua ve cung cac cot boolean / so can cho claim
# ----------------------------------------------------------------------------------------------
def compact(df, lab):
    n = df["Tot Fwd Pkts"] + df["Tot Bwd Pkts"]
    nf = df["Tot Fwd Pkts"]
    tcp, udp, p0 = df["Protocol"] == 6, df["Protocol"] == 17, df["Protocol"] == 0
    zp = (df["TotLen Fwd Pkts"] == 0) & (df["TotLen Bwd Pkts"] == 0)
    c = pd.DataFrame({
        "lab": np.asarray(lab),
        "port": df["Dst Port"].astype("int64").to_numpy(),
        "proto": df["Protocol"].astype("int64").to_numpy(),
        "zp": zp.to_numpy(),
        "flags01": df[FLAG_CNT + ["Fwd PSH Flags", "Fwd URG Flags"]].isin([0, 1]).all(axis=1).to_numpy(),
        "bulk0": (df[BULK] == 0).all(axis=1).to_numpy(),
        "bflag0": ((df["Bwd PSH Flags"] == 0) & (df["Bwd URG Flags"] == 0)).to_numpy(),
        "dbl": np.isclose(df["Pkt Size Avg"] * n, df["Pkt Len Mean"] * (n + 1), rtol=1e-4, atol=1e-2),
        "downup": np.isclose(df["Down/Up Ratio"], np.floor(df["Tot Bwd Pkts"] / nf.where(nf > 0)).fillna(0)),
        "dur120": (df["Flow Duration"] <= 120_000_000).to_numpy(),
        "p0ok": (~p0 | ((df["Dst Port"] == 0) & zp)).to_numpy(),
        "tcp_iwf_ok": (~tcp | (df["Init Fwd Win Byts"] != -1)).to_numpy(),
        "tcp_iwb_ok": (~(tcp & (df["Tot Bwd Pkts"] > 0)) | (df["Init Bwd Win Byts"] != -1)).to_numpy(),
        "udp_ok": (~udp | ((df[FLAG_CNT] == 0).all(axis=1) & (df["Init Fwd Win Byts"] == -1))).to_numpy(),
        "act_ok": (df["Fwd Act Data Pkts"] <= nf - 1).to_numpy(),
        "seg_ok": ((~tcp | df["Fwd Seg Size Min"].between(20, 60))
                   & (~udp | (df["Fwd Seg Size Min"] == 8))).to_numpy(),
        "iwf_ok": (~tcp | df["Init Fwd Win Byts"].between(0, 65535)).to_numpy(),
    })
    for a, b in PAIRS:
        c[f"eq__{a}__{b}"] = (df[a] == df[b]).to_numpy()
    for col, v in CONSTANTS.items():
        c[f"const__{col}"] = (df[col] == v).to_numpy()
    return c


# ----------------------------------------------------------------------------------------------
# Dinh nghia claim. value(khung da loc theo scope) -> so; test(so) -> bool
# ----------------------------------------------------------------------------------------------
def _frac(col):
    return lambda c: float(c[col].mean())


def _share(pred):
    return lambda c: float(pred(c).mean())


def claim_defs():
    D = []

    def add(cid, statement, kind, value, test, test_text, scope="rows"):
        D.append({"id": cid, "statement": statement, "kind": kind, "value": value, "test": test,
                  "test_text": test_text, "scope": scope})

    def one(v):
        return v == 1.0

    # --- bat bien theo dong ---------------------------------------------------------------------
    add("flags_binary", "The eight flag-count columns and Fwd PSH/URG Flags take only the values 0 and 1.",
        "invariant", _frac("flags01"), one, "= 1")
    add("bulk_zero", "The six bulk columns are 0 in every row.", "invariant", _frac("bulk0"), one, "= 1")
    add("bwd_dir_flags_zero", "Bwd PSH Flags and Bwd URG Flags are 0 in every row.", "invariant",
        _frac("bflag0"), one, "= 1")
    add("pkt_len_double_count", "Pkt Size Avg x (Tot Fwd Pkts + Tot Bwd Pkts) equals Pkt Len Mean x (Tot Fwd Pkts + "
        "Tot Bwd Pkts + 1) in every row.", "invariant", _frac("dbl"), one, "= 1")
    add("downup_integer", "Down/Up Ratio equals floor(Tot Bwd Pkts / Tot Fwd Pkts) in every row.", "invariant",
        _frac("downup"), one, "= 1")
    add("duration_le_120s", "Flow Duration never exceeds 120,000,000 microseconds.", "invariant",
        _frac("dur120"), one, "= 1")
    add("protocol0_port0_no_payload", "Every row with Protocol 0 has Dst Port 0 and no payload.", "invariant",
        _frac("p0ok"), one, "= 1")
    add("tcp_init_fwd_never_minus1", "No TCP row has Init Fwd Win Byts = -1.", "invariant",
        _frac("tcp_iwf_ok"), one, "= 1")
    add("tcp_with_bwd_init_bwd_never_minus1", "No TCP row with at least one backward packet has Init Bwd Win Byts = -1.",
        "invariant", _frac("tcp_iwb_ok"), one, "= 1")
    add("udp_flags_zero_init_fwd_minus1", "Every UDP row has all flag-count columns 0 and Init Fwd Win Byts = -1.",
        "invariant", _frac("udp_ok"), one, "= 1")
    add("act_data_le_fwd_minus1", "Fwd Act Data Pkts is at most Tot Fwd Pkts - 1 in every row.", "invariant",
        _frac("act_ok"), one, "= 1")
    add("seg_size_min_is_header", "Fwd Seg Size Min is between 20 and 60 for TCP rows and 8 for UDP rows.",
        "invariant", _frac("seg_ok"), one, "= 1")
    add("init_fwd_win_tcp_16bit", "Init Fwd Win Byts of TCP rows lies in [0, 65535], the range of a 16-bit field.",
        "invariant", _frac("iwf_ok"), one, "= 1")
    # --- cot trung / gan trung / cot hang: danh sach tim tren train (data_facts), kiem lai moi nguon --------
    for a, b in IDENT:
        add(f"identical__{a}__{b}", f"'{a}' equals '{b}' in every row.", "invariant",
            _frac(f"eq__{a}__{b}"), one, "= 1")
    for a, b, _ in NEAR:
        add(f"near_identical__{a}__{b}", f"'{a}' equals '{b}' in at least 99.9% of rows.", "threshold",
            _frac(f"eq__{a}__{b}"), lambda v: v >= 0.999, ">= 0.999")
    for col, v in CONSTANTS.items():
        add(f"constant__{col}", f"'{col}' is {v:g} in every row.", "invariant", _frac(f"const__{col}"), one, "= 1")
    # --- theo nhan ------------------------------------------------------------------------------
    for L in ["FTP_BruteForce", "DoS_attacks_SlowHTTPTest"]:
        add(f"twin_tcp21_nopayload__{L}", f"Every {L} flow goes to TCP port 21 and carries no payload.", "invariant",
            _share(lambda c: (c.proto == 6) & (c.port == 21) & c.zp), one, "= 1", scope=L)
    add("loic_udp_udp_port80", "Every DDOS_attack_LOIC_UDP flow is UDP to port 80.", "invariant",
        _share(lambda c: (c.proto == 17) & (c.port == 80)), one, "= 1", scope="DDOS_attack_LOIC_UDP")
    for L in ["DoS_attacks_GoldenEye", "DoS_attacks_Hulk", "DoS_attacks_Slowloris", "DDoS_attacks_LOIC_HTTP",
              "DDOS_attack_HOIC", "SQL_Injection"]:
        add(f"port80_all__{L}", f"Every {L} flow goes to port 80.", "invariant",
            _share(lambda c: c.port == 80), one, "= 1", scope=L)
    add("ssh_port22_ge999", "At least 99.9% of SSH_Bruteforce flows go to port 22.", "threshold",
        _share(lambda c: c.port == 22), lambda v: v >= 0.999, ">= 0.999", scope="SSH_Bruteforce")
    add("bot_tcp8080_gt95", "More than 95% of Bot flows go to TCP port 8080.", "threshold",
        _share(lambda c: (c.proto == 6) & (c.port == 8080)), lambda v: v > 0.95, "> 0.95", scope="Bot")
    add("xss_port80_ge90", "At least 90% of Brute_Force_XSS flows go to port 80.", "threshold",
        _share(lambda c: c.port == 80), lambda v: v >= 0.90, ">= 0.90", scope="Brute_Force_XSS")
    add("bfw_port80_top", "Port 80 is the most common destination port of Brute_Force_Web flows.", "top",
        lambda c: float(c.port.value_counts().idxmax()), lambda v: v == 80.0, "= 80", scope="Brute_Force_Web")
    add("bfw_has_udp_and_other_ports", "The Brute_Force_Web label contains UDP flows and flows to ports other than 80 "
        "(label level: training split, validation split, and the published CSV files taken together).", "exists",
        lambda c: float(bool((c.proto == 17).any()) and bool((c.port != 80).any())), one, "= 1",
        scope="Brute_Force_Web")
    add("infil_udp53_ge10", "At least 10% of Infiltration flows are UDP to port 53 (DNS).", "threshold",
        _share(lambda c: (c.proto == 17) & (c.port == 53)), lambda v: v >= 0.10, ">= 0.10", scope="Infiltration")
    add("infil_tcp443_ge10", "At least 10% of Infiltration flows are TCP to port 443 (HTTPS).", "threshold",
        _share(lambda c: (c.proto == 6) & (c.port == 443)), lambda v: v >= 0.10, ">= 0.10", scope="Infiltration")
    add("infil_max_port_lt50", "No single destination port carries half of the Infiltration flows.", "threshold",
        lambda c: float(c.port.value_counts(normalize=True).iloc[0]), lambda v: v < 0.5, "< 0.5", scope="Infiltration")
    return D


# ----------------------------------------------------------------------------------------------
# Tinh tren cac nguon
# ----------------------------------------------------------------------------------------------
def evaluate(defs, frames, kinds_exists=False):
    """frames: {ten nguon: khung gon}. Tra {id: {ten nguon: gia tri | None}}.
    kinds_exists=False: moi claim tru 'exists'; True: chi claim 'exists'."""
    out = {}
    for d in defs:
        if (d["kind"] == "exists") != kinds_exists:
            continue
        out[d["id"]] = {}
        for name, c in frames.items():
            g = c if d["scope"] == "rows" else c[c.lab == d["scope"]]
            # MIN_N chi ap cho ty le / nguong (it dong thi ty le dao dong); claim bat bien kiem tung dong nen
            # van tinh duoc voi nhan nho (vd FTP_BruteForce, SQL_Injection co < 20 dong trong val_full)
            small = len(g) == 0 or (d["scope"] != "rows" and d["kind"] != "invariant" and len(g) < MIN_N)
            out[d["id"]][name] = None if small else round(d["value"](g), 6)
    return out


def split_frames():
    frames = {}
    for name in ("train_full", "val_full"):
        df = pd.read_csv(Path(C.config.DATA_DIR) / f"{name}.csv", usecols=NEED + ["Label_Text"])
        frames[name.replace("_full", "")] = compact(df, df["Label_Text"])
        print(f"  {name}: {len(df)} dong", flush=True)
    return frames


def raw_frames(raw_dir):
    """{ngay: khung gon da dedup}, {ngay:h1|h2: cac nhan tan cong chia doi theo thu tu dong}, khung gop.
    Doc theo khuc, bo dong header lap va dong NaN/Inf, dedup trong ngay."""
    days, halves, pooled = {}, {}, []
    for path in sorted(Path(raw_dir).glob("*.csv")):
        tag = path.name.split("_")[0]
        parts, hashes, feats, nbad = [], [], None, 0
        for ch in pd.read_csv(path, chunksize=250_000, low_memory=False):
            ch.columns = [x.strip() for x in ch.columns]
            if feats is None:
                feats = [x for x in ch.columns
                         if x not in ("Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp", LABEL_COL_RAW)]
            ch = ch[ch[LABEL_COL_RAW].astype(str) != LABEL_COL_RAW]
            X = ch[feats].apply(pd.to_numeric, errors="coerce").astype("float64")
            ok = np.isfinite(X.to_numpy()).all(axis=1)
            nbad += int((~ok).sum())
            X, lab = X[ok], ch.loc[ok, LABEL_COL_RAW].map(std_label)
            hashes.append(pd.util.hash_pandas_object(pd.concat([X, lab.rename("L")], axis=1), index=False).to_numpy())
            parts.append(compact(X, lab))
        c = pd.concat(parts, ignore_index=True)
        _, first = np.unique(np.concatenate(hashes), return_index=True)
        c = c.iloc[np.sort(first)].reset_index(drop=True)
        days[tag] = c
        for L, g in c.groupby("lab", sort=False):
            k = len(g) // 2
            if L != "Benign" and k >= MIN_N:
                halves.setdefault(f"{tag}:h1", []).append(g.iloc[:k])
                halves.setdefault(f"{tag}:h2", []).append(g.iloc[k:])
        pooled.append(c[c.lab != "Benign"])
        print(f"  {tag}: {len(c)} dong sau dedup (bo {nbad} dong NaN/Inf)", flush=True)
    halves = {k: pd.concat(v, ignore_index=True) for k, v in halves.items()}
    return days, halves, pd.concat(pooled, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="doc 10 file CSV goc (lau, khoang 15-20 phut)")
    args = ap.parse_args()
    defs = claim_defs()
    old = json.loads(CLAIMS_JSON.read_text(encoding="utf-8")) if CLAIMS_JSON.exists() else {}

    frames = split_frames()
    values = evaluate(defs, frames)
    values.update(evaluate(defs, frames, kinds_exists=True))
    if args.raw:
        bench = os.environ.get("UTS_BENCH")
        if not bench:
            raise SystemExit("--raw can bien moi truong UTS_BENCH (thu muc zeroshot_benchmark)")
        days, halves, pooled = raw_frames(Path(bench) / "01_raw_data" / "raw_csv")
        for part in (evaluate(defs, days), evaluate([d for d in defs if d["scope"] != "rows"], halves),
                     evaluate(defs, {"raw_pooled": pooled}, kinds_exists=True)):
            for cid, v in part.items():
                values[cid].update(v)
    else:
        # giu gia tri file goc tu lan --raw truoc; claim moi chua co gia tri goc -> holds_all = false
        for cid in values:
            for src, v in old.get(cid, {}).get("values", {}).items():
                if src not in ("train", "val"):
                    values[cid].setdefault(src, v)

    out = {}
    for d in defs:
        vals = values[d["id"]]
        present = {k: v for k, v in vals.items() if v is not None}
        has_raw = any(k not in ("train", "val") for k in present)
        failing = sorted(k for k, v in present.items() if not d["test"](v))
        holds = "train" in present and "val" in present and has_raw and not failing
        out[d["id"]] = {"statement": d["statement"], "kind": d["kind"], "scope": d["scope"], "test": d["test_text"],
                        "values": vals, "holds_all": holds, "failing": failing}
    C.write_json(CLAIMS_JSON, out)
    bad = [k for k, v in out.items() if not v["holds_all"]]
    print(f"-> {CLAIMS_JSON}: {len(out)} claim, {len(out) - len(bad)} dung o moi nguon")
    for k in bad:
        print(f"   KHONG DUNG / CHUA DU NGUON: {k}  failing={out[k]['failing']}")


if __name__ == "__main__":
    main()
