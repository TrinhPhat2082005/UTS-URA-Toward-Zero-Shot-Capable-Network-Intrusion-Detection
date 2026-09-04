import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import seaborn as sns
from scapy.all import rdpcap, IP, TCP, UDP, ICMP

rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "axes.titleweight":  "bold",
    "axes.labelweight":  "bold",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":      "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linestyle":    "--",
    "grid.linewidth":     0.5,
    "grid.color":        "#cccccc",
    "grid.alpha":         0.7,
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "axes.edgecolor":    "#333333",
    "text.color":        "#1a1a1a",
    "axes.labelcolor":   "#1a1a1a",
    "xtick.color":       "#1a1a1a",
    "ytick.color":       "#1a1a1a",
})

TAB10 = plt.get_cmap("tab10").colors


class PcapVisualizer:
    """Read .pcap files and generate standard network traffic analysis charts."""

    def __init__(self, pcap_path: str,
                 output_dir: str = "project_root/data/vlm_dataset/chart"):
        self.pcap_path  = pcap_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"[Visualizer] Reading PCAP: {pcap_path}")
        packets = rdpcap(pcap_path)
        print(f"[Visualizer] Loaded {len(packets)} packets.")

        records = []
        for pkt in packets:
            if not pkt.haslayer(IP):
                continue
            ip  = pkt[IP]
            proto = "Other"
            src_port = dst_port = 0
            if pkt.haslayer(TCP):
                proto    = "TCP"
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif pkt.haslayer(UDP):
                proto    = "UDP"
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
            elif pkt.haslayer(ICMP):
                proto = "ICMP"
            records.append({
                "timestamp": float(pkt.time),
                "src_ip":    ip.src,
                "dst_ip":    ip.dst,
                "src_port":  src_port,
                "dst_port":  dst_port,
                "protocol":  proto,
                "length":    len(pkt),
            })

        self.df = pd.DataFrame(records)
        if not self.df.empty:
            self.df["datetime"] = pd.to_datetime(self.df["timestamp"], unit="s")
            self.df = self.df.sort_values("timestamp").reset_index(drop=True)
        print(f"[Visualizer] Extracted {len(self.df)} IP packets.")

    def _finalize(self, fig, out_path: str):
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[Visualizer] Saved: {out_path}")

    def plot_traffic_volume(self, resample_interval: str = "1s",
                            filename: str = "chart_traffic_volume.png") -> str:
        if self.df.empty:
            print("[Warning] No data for Traffic Volume plot.")
            return ""

        ts           = self.df.set_index("datetime")
        bytes_s      = ts["length"].resample(resample_interval).sum().fillna(0)
        pkts_s       = ts["length"].resample(resample_interval).count().fillna(0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle("Network Traffic Volume Over Time", fontsize=13, fontweight="bold", y=1.01)

        c1, c2 = TAB10[0], TAB10[1]

        ax1.fill_between(bytes_s.index, bytes_s.values, alpha=0.25, color=c1)
        ax1.plot(bytes_s.index, bytes_s.values, color=c1, linewidth=1.4, label="Bytes/interval")
        peak_idx = bytes_s.idxmax()
        ax1.annotate(
            f"Peak: {bytes_s.max()/1024:.1f} KB",
            xy=(peak_idx, bytes_s.max()),
            xytext=(10, -18), textcoords="offset points",
            fontsize=8, color=c1,
            arrowprops=dict(arrowstyle="->", color=c1, lw=0.8),
        )
        ax1.set_ylabel("Bytes per Interval")
        ax1.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x/1024:.0f} KB" if x >= 1024 else f"{x:.0f} B"))
        ax1.legend(loc="upper right")

        ax2.fill_between(pkts_s.index, pkts_s.values, alpha=0.25, color=c2)
        ax2.plot(pkts_s.index, pkts_s.values, color=c2, linewidth=1.4, label="Packets/interval")
        ax2.set_ylabel("Packet Count per Interval")
        ax2.set_xlabel("Time")
        ax2.legend(loc="upper right")

        fig.autofmt_xdate(rotation=25)

        out = os.path.join(self.output_dir, filename)
        self._finalize(fig, out)
        return out

    def plot_protocol_distribution(
            self, filename: str = "chart_protocol_distribution.png") -> str:
        if self.df.empty:
            print("[Warning] No data for Protocol Distribution plot.")
            return ""

        proto_counts = self.df["protocol"].value_counts()
        proto_bytes  = self.df.groupby("protocol")["length"].sum().reindex(proto_counts.index)
        colors       = list(TAB10[:len(proto_counts)])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Protocol Distribution Analysis", fontsize=13, fontweight="bold")

        explode = [0.05 if i == 0 else 0 for i in range(len(proto_counts))]
        wedges, texts, autotexts = ax1.pie(
            proto_counts.values,
            labels=proto_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
            explode=explode,
            textprops={"fontsize": 9},
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            pctdistance=0.78,
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_fontweight("bold")
        ax1.set_title("(a) By Packet Count")
        ax1.grid(False)

        bars = ax2.barh(proto_bytes.index, proto_bytes.values,
                        color=colors, edgecolor="white", height=0.55)
        ax2.set_xlabel("Total Bytes Transferred")
        ax2.set_title("(b) By Total Bytes")
        ax2.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x/1024:.0f} KB" if x >= 1024 else f"{x:.0f} B"))
        max_val = proto_bytes.max()
        for bar, val in zip(bars, proto_bytes.values):
            label = f"{val/1024:.1f} KB" if val >= 1024 else str(val)
            ax2.text(bar.get_width() + max_val * 0.02,
                     bar.get_y() + bar.get_height() / 2,
                     label, va="center", fontsize=8)
        ax2.invert_yaxis()

        out = os.path.join(self.output_dir, filename)
        self._finalize(fig, out)
        return out

    def plot_src_dst_flow(self, top_n: int = 15,
                          filename: str = "chart_src_dst_flow.png") -> str:
        if self.df.empty:
            print("[Warning] No data for Source-Destination Flow plot.")
            return ""

        flow = (self.df
                .groupby(["src_ip", "dst_ip"])
                .agg(packets=("length", "count"), total_bytes=("length", "sum"))
                .reset_index()
                .sort_values("total_bytes", ascending=False)
                .head(top_n))
        flow["label"] = flow["src_ip"] + "  →  " + flow["dst_ip"]

        fig, ax = plt.subplots(figsize=(11, max(5, top_n * 0.42)))
        fig.suptitle(f"Top {top_n} Source–Destination Flows by Traffic Volume",
                     fontsize=13, fontweight="bold")

        norm   = plt.Normalize(flow["total_bytes"].min(), flow["total_bytes"].max())
        cmap   = plt.get_cmap("Blues")
        colors = [cmap(0.35 + 0.65 * norm(v)) for v in flow["total_bytes"].values]

        bars = ax.barh(flow["label"], flow["total_bytes"],
                       color=colors, edgecolor="#888888", linewidth=0.5, height=0.65)

        ax.set_xlabel("Total Bytes Transferred")
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x/1024:.0f} KB" if x >= 1024 else f"{x:.0f}"))

        max_val = flow["total_bytes"].max()
        for bar, (_, row) in zip(bars, flow.iterrows()):
            ax.text(bar.get_width() + max_val * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{row["packets"]} pkts',
                    va="center", fontsize=7.5)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(flow["total_bytes"].min(), max_val))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.02)
        cbar.set_label("Total Bytes", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=8)

        out = os.path.join(self.output_dir, filename)
        self._finalize(fig, out)
        return out

    def plot_port_activity(self, top_n: int = 20,
                           filename: str = "chart_port_activity.png") -> str:
        if self.df.empty:
            print("[Warning] No data for Port Activity plot.")
            return ""

        tcp_udp = self.df[self.df["protocol"].isin(["TCP", "UDP"])].copy()
        if tcp_udp.empty:
            print("[Warning] No TCP/UDP packets for Port Activity plot.")
            return ""

        WELL_KNOWN = {
            20: "FTP-D", 21: "FTP", 22: "SSH", 23: "Telnet",
            25: "SMTP",  53: "DNS", 80: "HTTP", 110: "POP3",
            143: "IMAP", 443: "HTTPS", 993: "IMAPS", 995: "POP3S",
            3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL",
            8080: "HTTP-Alt", 8443: "HTTPS-Alt",
        }

        src_ports = (tcp_udp.groupby("src_port")["length"]
                     .agg(packets="count", bytes="sum")
                     .reset_index().rename(columns={"src_port": "port"}))
        src_ports["direction"] = "Source Port"

        dst_ports = (tcp_udp.groupby("dst_port")["length"]
                     .agg(packets="count", bytes="sum")
                     .reset_index().rename(columns={"dst_port": "port"}))
        dst_ports["direction"] = "Destination Port"

        combined  = pd.concat([src_ports, dst_ports], ignore_index=True)
        combined["port"] = combined["port"].astype(int)
        top_ports = (combined.groupby("port")["packets"].sum()
                     .nlargest(top_n).index.tolist())
        plot_data = combined[combined["port"].isin(top_ports)].copy()

        labels = [f"{p} ({WELL_KNOWN[p]})" if p in WELL_KNOWN else str(p)
                  for p in top_ports]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.42)))
        fig.suptitle(f"Top {top_n} Port Activity: Source vs. Destination",
                     fontsize=13, fontweight="bold")

        color_map = {"Source Port": TAB10[0], "Destination Port": TAB10[3]}
        bar_h     = 0.35
        positions = np.arange(len(top_ports))

        for i, direction in enumerate(["Source Port", "Destination Port"]):
            subset = plot_data[plot_data["direction"] == direction]
            vals   = [subset[subset["port"] == p]["packets"].sum() for p in top_ports]
            offset = (i - 0.5) * bar_h
            ax.barh(positions + offset, vals, height=bar_h,
                    color=color_map[direction], label=direction,
                    edgecolor="white", linewidth=0.6, alpha=0.9)

        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel("Packet Count")
        ax.invert_yaxis()

        for y in positions:
            ax.axhline(y=y, color="#dddddd", linewidth=0.4, zorder=0)

        ax.legend(loc="lower right", framealpha=0.9)

        out = os.path.join(self.output_dir, filename)
        self._finalize(fig, out)
        return out

    def generate_all(self, resample: str = "1s",
                     top_n_flow: int = 15,
                     top_n_port: int = 20) -> list:
        print("[Visualizer] Generating core analysis charts...")
        paths = [
            self.plot_traffic_volume(resample_interval=resample),
            self.plot_protocol_distribution(),
            self.plot_src_dst_flow(top_n=top_n_flow),
            self.plot_port_activity(top_n=top_n_port),
        ]
        valid = [p for p in paths if p]
        print(f"[Visualizer] Completed: {len(valid)}/4 charts saved.")
        return valid


if __name__ == "__main__":
    pcap_dir   = "project_root/data/raw_pcap"
    pcap_files = glob.glob(os.path.join(pcap_dir, "*.pcap"))
    if not pcap_files:
        print(f"[Error] No .pcap files found in {pcap_dir}")
    else:
        pcap_files.sort(key=os.path.getsize)
        test_pcap = pcap_files[0]
        print(f"[Visualizer] Testing with: {test_pcap}")
        viz = PcapVisualizer(
            pcap_path=test_pcap,
            output_dir="project_root/data/vlm_dataset/chart"
        )
        results = viz.generate_all()
        print("\n>>> Output files:")
        for p in results:
            print(f"     {p}")
