"""
H2 — Giao thuc, cong, co TCP: dinh nghia chuan (IANA, IETF), khong lien quan toi dataset.

  - H2-PROTO-6 / -17: dong dang ky IANA Protocol Numbers (rendered). Chi hai so nay vi CICFlowMeter-V3 chi ghi 6
    (TCP) hoac 17 (UDP); con lai la gia tri mac dinh 0 (xem H3-NOTRANSPORT) — trich ma nguon trong support.
  - H2-PORT-<p>: MOI cong co dang ky ten dich vu trong IANA Service Name and Transport Protocol Port Number
    Registry (tcp/udp), ghep tu dong tu cac dong cua registry; khong co buoc chon loc.
  - H2-PORT-NOTE: cau nguyen van RFC 6335 sec. 9 (dang ky cong khong cho biet flow dung dich vu nao);
    retrieve.py dua kem moi khi chon mot doan cong.
  - H2-FLAG-<F>: dinh nghia 8 co trong RFC 9293 muc 3.1, NGUYEN VAN; H2-FLAG-ECN-SETUP: RFC 3168 (SYN co ca ECE
    va CWR). retrieve.py chon doan cua co MA COT THUC SU CHUA (flag_order.flag_column_holds), khong theo ten cot.

    khoa {"protocol": 6} | {"port": 80} | {"flag": "RST"}
"""
from __future__ import annotations

import csv
import io
from collections import defaultdict

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
PORT_NOTE_QUOTE = ("the fact that network traffic is flowing to or from an assigned port number does not mean that it "
                   "is \"good\" traffic, or even that it is used by the assigned service.")
PROTOCOL_CODE = [
    {"src": "cicflowmeter-src-2018", "locator": "PacketReader.java, TCP branch", "quote": "packetInfo.setProtocol(6);"},
    {"src": "cicflowmeter-src-2018", "locator": "PacketReader.java, UDP branch", "quote": "packetInfo.setProtocol(17);"},
]


def _iana_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def _row_quote(fields):
    """Cac o dau cua mot dong IANA, ghi lai DUNG nhu trong file CSV (o co dau phay / xuong dong nam trong "...")."""
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(fields)
    return buf.getvalue()


def _port_passages():
    rows = [r for r in _iana_rows(C.source_path("iana-ports", "service-names-port-numbers.csv"))[1:]
            if len(r) > 3 and r[0] and r[1].isdigit() and r[2] in ("tcp", "udp")]
    by_port = defaultdict(list)
    for r in rows:
        by_port[int(r[1])].append(r)
    out = []
    for port in sorted(by_port):
        names = {}                                    # ten dich vu -> (transport, mo ta), giu thu tu registry
        for r in by_port[port]:
            entry = names.setdefault(r[0], {"transports": [], "desc": r[3]})
            if r[2] not in entry["transports"]:
                entry["transports"].append(r[2])
        # mo ta IANA co the co xuong dong va dau cham cuoi: gop khoang trang, bo dau cham cuoi (support giu nguyen dong)
        services = "; ".join(f"'{n}' ({', '.join(e['transports'])})"
                             + (f": {C.normalize(e['desc']).rstrip('.')}" if e["desc"] else "")
                             for n, e in names.items())
        out.append(C.passage(f"H2-PORT-{port}", "H2", {"port": port}, "rendered",
                             f"Port {port} is registered by IANA as {services}.",
                             title=f"Port {port} ({next(iter(names))})",
                             support=[{"src": "iana-ports", "locator": f"service-names-port-numbers.csv, {r[0]} {port}/{r[2]}",
                                       "quote": _row_quote(r[:4])} for r in by_port[port]]))
    return out


def build(ctx):
    out = []
    # --- giao thuc ------------------------------------------------------------------------------
    for row in _iana_rows(C.source_path("iana-protocols", "protocol-numbers-1.csv")):
        if row and row[0] in ("6", "17"):
            num, kw, name, ref = row[0], row[1], row[2], row[4]
            out.append(C.passage(f"H2-PROTO-{num}", "H2", {"protocol": int(num)}, "rendered",
                                 f"IP protocol number {num} is {kw} ({name}), specified in {ref.split(';')[0]}.",
                                 title=f"Protocol {num} ({kw})",
                                 support=[{"src": "iana-protocols", "locator": f"protocol-numbers-1.csv, row {num}",
                                           "quote": _row_quote(row[:5])}] + PROTOCOL_CODE))
    # --- cong -----------------------------------------------------------------------------------
    out += _port_passages()
    out.append(C.passage("H2-PORT-NOTE", "H2", {"port": "note"}, "verbatim", PORT_NOTE_QUOTE,
                         title="What a registered port does not tell",
                         source={"src": "rfc6335", "locator": "Section 9, Security Considerations"}))
    # --- co TCP ---------------------------------------------------------------------------------
    for flag, quote in FLAG_QUOTES.items():
        out.append(C.passage(f"H2-FLAG-{flag}", "H2", {"flag": flag}, "verbatim", quote, title=f"TCP flag {flag}",
                             source={"src": "rfc9293", "locator": "Section 3.1, Header Format, Control bits"}))
    out.append(C.passage("H2-FLAG-ECN-SETUP", "H2", {"flag": "ECE+CWR"}, "verbatim", ECN_SETUP_QUOTE,
                         title="ECN-setup SYN (ECE and CWR set together)",
                         source={"src": "rfc3168", "locator": "Section 6.1.1"}))
    return out
