"""
Flow-level replacement for project_root/core/layer2_vlm/visualizer.py's
PcapVisualizer, used only by the VLM confidence-gate experiment.

PcapVisualizer needs packet-level records (timestamp, src/dst IP, port) read
from a .pcap file. The FP/TP test sets for this experiment come from
UTS_Layer1/binary_split_dataset_z_Infilteration/{test,val}_binary.csv, which
are CICFlowMeter FLOW-level aggregates (one row per flow, no per-packet
timeline, and Src IP/Dst IP are NaN in this CIC-IDS-2018 split) — there is no
PCAP to read, so PcapVisualizer's charts cannot be reproduced as-is.

This module renders 4 different charts, computed only from fields present in
a single flow-level CSV row, to fill the same "visual evidence" slot the VLM
prompt expects. They answer analogous questions using flow-aggregate
statistics instead of packet timelines:
  1. Packet length profile   (Fwd/Bwd min/mean/max)      -- vs. traffic volume-over-time
  2. Flag composition        (SYN/ACK/RST/FIN/PSH/URG)   -- vs. protocol distribution
  3. Inter-arrival timing    (Flow/Fwd/Bwd IAT mean+std)  -- vs. src-dst flow chart
  4. Directional byte/packet balance (Fwd vs Bwd, Down/Up Ratio) -- vs. port activity

This is a deliberate scope difference from the production pipeline and must
be stated as such in the paper's Methodology section.
"""
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 110,
    "savefig.bbox": "tight",
})

TAB10 = plt.get_cmap("tab10").colors


def _finalize(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_packet_length_profile(row, out_dir, filename="chart_pkt_length_profile.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = ["Fwd", "Bwd"]
    stats = ["Min", "Mean", "Max"]
    x = range(len(stats))
    width = 0.35
    for i, g in enumerate(groups):
        vals = [max(float(row.get(f"{g} Pkt Len {s}", 0.0)), 0.0) for s in stats]
        offset = (i - 0.5) * width
        ax.bar([xi + offset for xi in x], vals, width=width, label=g, color=TAB10[i])
    ax.set_xticks(list(x))
    ax.set_xticklabels(stats)
    ax.set_ylabel("Packet Length (bytes)")
    ax.set_title("Forward vs. Backward Packet Length Profile")
    ax.legend()
    return _finalize(fig, os.path.join(out_dir, filename))


_FLAG_HOLDS = None


def _flag_holds():
    """{cot CSV: co ma cot THUC SU chua} theo CICFlowMeter-V3 2018 (thu tu HashMap), doc tu KB
    (knowledge_base/store/index.json, sinh boi build/flag_order.py). Vd 'SYN Flag Cnt' chua PSH."""
    global _FLAG_HOLDS
    if _FLAG_HOLDS is None:
        import json
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "knowledge_base", "store", "index.json")
        with open(p, encoding="utf-8") as f:
            _FLAG_HOLDS = json.load(f)["flag_column_holds"]
    return _FLAG_HOLDS


FLAG_ORDER = ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "CWR", "ECE"]


def plot_flag_composition(row, out_dir, filename="chart_flag_composition.png"):
    """8 co TCP cua GOI DAU TIEN (ban 2018 chi doc co o goi dau; gia tri 0/1). Nhan thanh = co ma cot thuc su
    chua, khong phai ten cot (ten cot bi hoan vi), de anh khop voi dinh nghia H1/H2 cua KB."""
    holds = _flag_holds()
    col_of = {flag: col for col, flag in holds.items() if col.endswith(("Flag Cnt", "Flag Count"))}
    vals = [max(float(row.get(col_of[f], 0.0)), 0.0) for f in FLAG_ORDER]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [TAB10[i % 10] for i in range(len(FLAG_ORDER))]
    ax.bar(FLAG_ORDER, vals, color=colors, edgecolor="white")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 1])
    ax.set_ylabel("Set in the first packet (0/1)")
    ax.set_title("TCP Flags of the First Packet (this flow)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    return _finalize(fig, os.path.join(out_dir, filename))


def plot_iat_timing(row, out_dir, filename="chart_iat_timing.png"):
    groups = ["Flow", "Fwd", "Bwd"]
    means = [max(float(row.get(f"{g} IAT Mean", 0.0)), 0.0) for g in groups]
    stds = [max(float(row.get(f"{g} IAT Std", 0.0)), 0.0) for g in groups]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(groups))
    ax.bar(x, means, yerr=stds, capsize=4, color=TAB10[2], edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{g} IAT" for g in groups])
    ax.set_ylabel("Inter-Arrival Time (µs), mean ± std")
    ax.set_title("Inter-Arrival Timing by Direction")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    return _finalize(fig, os.path.join(out_dir, filename))


def plot_directional_balance(row, out_dir, filename="chart_directional_balance.png"):
    fwd_pkts = max(float(row.get("Tot Fwd Pkts", 0.0)), 0.0)
    bwd_pkts = max(float(row.get("Tot Bwd Pkts", 0.0)), 0.0)
    fwd_bytes = max(float(row.get("TotLen Fwd Pkts", 0.0)), 0.0)
    bwd_bytes = max(float(row.get("TotLen Bwd Pkts", 0.0)), 0.0)
    dst_port = row.get("Dst Port", "?")
    protocol = row.get("Protocol", "?")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(f"Directional Balance — Dst Port {dst_port} (Proto {protocol})",
                 fontsize=12, fontweight="bold")

    def _safe_pie(ax, vals, title):
        if sum(vals) <= 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return
        ax.pie(vals, labels=["Fwd", "Bwd"], autopct="%1.1f%%",
               colors=[TAB10[0], TAB10[3]], startangle=90)
        ax.set_title(title)

    _safe_pie(ax1, [fwd_pkts, bwd_pkts], "Packet Count")
    _safe_pie(ax2, [fwd_bytes, bwd_bytes], "Byte Volume")

    return _finalize(fig, os.path.join(out_dir, filename))


def generate_flow_charts(row, out_dir):
    """row: a dict-like (pandas Series or dict) for a single flow, using
    CIC-IDS-2018 Title Case column names. Returns list of 4 PNG paths."""
    os.makedirs(out_dir, exist_ok=True)
    return [
        plot_packet_length_profile(row, out_dir),
        plot_flag_composition(row, out_dir),
        plot_iat_timing(row, out_dir),
        plot_directional_balance(row, out_dir),
    ]
