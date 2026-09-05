import subprocess
import os
import sys
from datetime import datetime

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_DIR      = os.path.join(BASE_DIR, "data", "processed_csv")
PCAP_DIR     = os.path.join(BASE_DIR, "data", "raw_pcap")

class LiveSniffer:
    """Network traffic sniffer using TShark with automated CICFlowMeter extraction."""
    
    def __init__(self, interface: str = "Wi-Fi", keep_pcap: bool = False):
        """Initialize sniffer."""
        self.interface = interface
        self.keep_pcap = keep_pcap
        # Output directory ensures
        os.makedirs(CSV_DIR, exist_ok=True)
        os.makedirs(PCAP_DIR, exist_ok=True)

    def start_capture(self, window_duration: int = 10):
        """Start continuous capture in time window snapshots."""
        print(f"[Sniffer] Interface: {self.interface} | Architecture: Hybrid (TShark -> CICFlowMeter)")
        print(f"[Sniffer] Window duration: {window_duration}s | Keep PCAP: {self.keep_pcap}")
        try:
            while True:
                pcap, csv = self._capture_and_extract(window_duration)
                yield pcap, csv
        except KeyboardInterrupt:
            print("[Sniffer] Process terminated.")

    def _capture_and_extract(self, window_duration: int):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pcap_file = os.path.join(PCAP_DIR, f"capture_{timestamp}.pcap")
        csv_file  = os.path.join(CSV_DIR, f"flow_{timestamp}.csv")
        
        # 1. Capture using TShark (Raw PCAP)
        cmd_pcap = [
            "tshark",
            "-i", self.interface,
            "-a", f"duration:{window_duration}",
            "-w", pcap_file,      
        ]
        
        print(f"\n[Capture] Listening for {window_duration}s -> {os.path.basename(pcap_file)}")
        subprocess.run(cmd_pcap, stderr=subprocess.DEVNULL)
        
        # Validate PCAP
        if not os.path.exists(pcap_file) or os.path.getsize(pcap_file) == 0:
            if os.path.exists(pcap_file):
                os.remove(pcap_file)
            print("[Capture] No traffic captured.")
            return None, None

        # 2. Extract features using CICFlowMeter
        print(f"[Extract] Running CICFlowMeter on {os.path.basename(pcap_file)}...")
        
        # Robustly find cicflowmeter executable (especially for Windows venv)
        cic_cmd = "cicflowmeter"
        python_scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
        if os.path.exists(os.path.join(python_scripts_dir, "cicflowmeter.exe")):
            cic_cmd = os.path.join(python_scripts_dir, "cicflowmeter.exe")
        elif os.path.exists(os.path.join(os.path.dirname(sys.executable), "cicflowmeter")):
            cic_cmd = os.path.join(os.path.dirname(sys.executable), "cicflowmeter")

        # Call cicflowmeter as a python module for maximum compatibility
        cmd_cic = [
            sys.executable, "-m", "cicflowmeter.sniffer",
            "-f", pcap_file,
            "-c", csv_file
        ]
        
        # Run process
        try:
            # Capture output to debug why it fails with 1
            result = subprocess.run(cmd_cic, capture_output=True, text=True)
            
            # Validate output CSV
            if result.returncode == 0 and os.path.exists(csv_file):
                size_kb = os.path.getsize(csv_file) / 1024
                print(f"[OK] Extracted features to {os.path.basename(csv_file)} ({size_kb:.1f} KB)")
            else:
                print(f"[Error] CICFlowMeter failed (Exit Code: {result.returncode})")
                if result.stderr:
                    print(f"[Debug] Error info: {result.stderr.strip()}")
        except FileNotFoundError:
            print("[Critical Error] 'cicflowmeter' command not found.")
            return pcap_file, None 
            
        # 3. Auto-Cleanup
        if not self.keep_pcap:
            try:
                if os.path.exists(pcap_file):
                    os.remove(pcap_file)
                    print(f"[Cleanup] Removed temporary PCAP: {os.path.basename(pcap_file)}")
            except OSError as e:
                print(f"[Warning] Could not remove PCAP: {e}")

        return pcap_file, csv_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LiveSniffer — Hybrid TShark & CICFlowMeter")
    parser.add_argument("-i", "--interface", default="Wi-Fi",  help="Tên network interface")
    parser.add_argument("-w", "--window",    default=10, type=int, help="Độ dài cửa sổ thời gian (giây), mặc định 10")
    parser.add_argument("--keep-pcap", action="store_true", help="Gắn flag này để không xóa file PCAP sau khi chạy xong")
    args = parser.parse_args()

    sniffer = LiveSniffer(interface=args.interface, keep_pcap=args.keep_pcap)
    sniffer.start_capture(window_duration=args.window)
