"""
Cot co nao trong CSV cua CSE-CIC-IDS2018 THUC SU chua co TCP nao (dinh nghia co theo phien ban CICFlowMeter-V3
ma dataset dung, ma nguon @50d668db; bang chung trich dan nam o definitions/feature_notes.yaml, XN-FLAG-ORDER).

CICFlowMeter 2018 giu 8 bo dem co trong java.util.HashMap, chen theo thu tu FIN, SYN, RST, PSH, ACK, URG, CWR,
ECE, roi ghi CSV bang vong `for (key : flagCounts.keySet())`; tieu de cot thi co dinh theo ten. Thu tu duyet cua
HashMap (OpenJDK 8: hash = h ^ (h >>> 16), bucket = hash & (n - 1), String.hashCode = 31*h + c) khong phai thu
tu chen, nen gia tri bi ghi lech so voi ten cot. Ham duoi tinh lai dung thu tu do. Commit 6833ccd4 (06/2018)
cua chinh CICFlowMeter bo vong lap nay va ghi co theo thu tu tuong minh.
"""
from __future__ import annotations

FLAG_KEYS = ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "CWR", "ECE"]          # thu tu put() trong initFlags()
FLAG_COLUMNS = ["FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",  # thu tu tieu de CSV (//50..57)
                "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt"]
FLAG_OF_NAME = {"FIN Flag Cnt": "FIN", "SYN Flag Cnt": "SYN", "RST Flag Cnt": "RST", "PSH Flag Cnt": "PSH",
                "ACK Flag Cnt": "ACK", "URG Flag Cnt": "URG", "CWE Flag Count": "CWR", "ECE Flag Cnt": "ECE"}
DIRECTIONAL = {"Fwd PSH Flags": "PSH", "Fwd URG Flags": "URG"}                  # ghi tuong minh, khong lech


def java_hashmap_order(keys, capacity=16):
    """Thu tu duyet cua java.util.HashMap (OpenJDK 8) voi khoa chuoi chen theo thu tu `keys` (8 khoa < 12 = 0.75*16
    nen capacity giu 16)."""
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


def flag_column_holds():
    """{ten cot CSV: co ma cot do thuc su chua}; gom ca Fwd PSH/URG Flags (khong lech)."""
    return {**dict(zip(FLAG_COLUMNS, java_hashmap_order(FLAG_KEYS))), **DIRECTIONAL}


if __name__ == "__main__":
    for col, flag in flag_column_holds().items():
        print(f"{col:16s} -> {flag}")
