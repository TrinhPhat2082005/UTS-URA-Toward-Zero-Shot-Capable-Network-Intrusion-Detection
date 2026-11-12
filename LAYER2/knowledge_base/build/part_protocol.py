"""
H2 — Giao thuc, cong, co TCP: dinh nghia chuan, dung chung moi nhan.

Y tuong: muc 2 cua report mo ta hanh vi quan sat duoc (cong dich, giao thuc, co); VLM can nghia chuan
cua chung de khong doan.
  - H2-PROTO-6 / -17: dong dang ky IANA Protocol Numbers (rendered tu cac truong nguyen van).
    Protocol 0 KHONG co doan H2: trong CICFlowMeter no la gia tri mac dinh (xem H3-NOTRANSPORT).
  - H2-PORT-<p>: cong chinh (chiem >= 1% flow cua mot nhan tan cong, data_facts.attack_main_ports),
    ten dich vu va mo ta tu IANA Port Registry. Cong 0 cung do H3-NOTRANSPORT giai thich.
  - H2-FLAG-<F>: dinh nghia 8 co trong RFC 9293 muc 3.1, NGUYEN VAN; H2-FLAG-ECN-SETUP: RFC 3168 (SYN co
    ca ECE va CWR). Khi mot cot co nam trong top feature, retrieve.py chon doan cua co MA COT DO THUC SU
    CHUA (data_facts.flag_column_holds), khong phai co trong ten cot.

    khoa {"protocol": 6} | {"port": 80} | {"flag": "RST"}
"""
from __future__ import annotations

import csv

import common as C

FLAG_QUOTES = {
    "CWR": "CWR: 1 bit Congestion Window Reduced (see [6]).",
    "ECE": "ECE: 1 bit ECN-Echo (see [6]).",
    "URG": "URG: 1 bit Urgent pointer field is significant.",
    "ACK": "ACK: 1 bit Acknowledgment field is significant.",
    "PSH": "PSH: 1 bit Push function (see the Send Call description in Section 3.9.1).",
    "RST": "RST: 1 bit Reset the connection.",
    "SYN": "SYN: 1 bit Synchronize sequence numbers.",
    "FIN": "FIN: 1 bit No more data from sender.",
}
ECN_SETUP_QUOTE = ('We call a SYN packet with the ECE and CWR flags set an "ECN-setup SYN packet", and we call a SYN '
                   'packet with at least one of the ECE and CWR flags not set a "non-ECN-setup SYN packet".')


def _iana_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def build(ctx):
    out = []
    # --- giao thuc ------------------------------------------------------------------------------
    for row in _iana_rows(C.source_path("iana-protocols", "protocol-numbers-1.csv")):
        if row and row[0] in ("6", "17"):
            num, kw, name, ref = row[0], row[1], row[2], row[4]
            quote = ",".join(row[:5])
            out.append(C.passage(f"H2-PROTO-{num}", "H2", {"protocol": int(num)}, "rendered",
                                 f"IP protocol number {num} is {kw} ({name}), specified in {ref.split(';')[0]}.",
                                 title=f"Protocol {num} ({kw})",
                                 support=[{"src": "iana-protocols", "locator": f"protocol-numbers-1.csv, row {num}",
                                           "quote": quote}]))
    # --- cong -----------------------------------------------------------------------------------
    rows = _iana_rows(C.source_path("iana-ports", "service-names-port-numbers.csv"))
    for port in ctx["facts"]["attack_main_ports"]["ports"]:
        if port == "0":
            continue
        hit = [r for r in rows if len(r) > 3 and r[1] == port and r[0] and r[2] in ("tcp", "udp")]
        if not hit:
            continue
        r = hit[0]
        quote = ",".join(r[:4])
        out.append(C.passage(f"H2-PORT-{port}", "H2", {"port": int(port)}, "rendered",
                             f"Port {port} is registered by IANA as service '{r[0]}': {r[3]}.",
                             title=f"Port {port} ({r[0]})",
                             support=[{"src": "iana-ports", "locator": f"service-names-port-numbers.csv, {r[0]} {port}/{r[2]}",
                                       "quote": quote},
                                      {"data": f"attack_main_ports.ports.{port}"}]))
    # --- co TCP ---------------------------------------------------------------------------------
    for flag, quote in FLAG_QUOTES.items():
        out.append(C.passage(f"H2-FLAG-{flag}", "H2", {"flag": flag}, "verbatim", quote,
                             title=f"TCP flag {flag}",
                             source={"src": "rfc9293", "locator": "Section 3.1, Header Format, Control bits"}))
    out.append(C.passage("H2-FLAG-ECN-SETUP", "H2", {"flag": "ECE+CWR"}, "verbatim", ECN_SETUP_QUOTE,
                         title="ECN-setup SYN (ECE and CWR set together)",
                         source={"src": "rfc3168", "locator": "Section 6.1.1"}))
    return out
