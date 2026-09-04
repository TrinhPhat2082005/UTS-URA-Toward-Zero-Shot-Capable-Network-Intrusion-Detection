"""
clear_buffer.py - Pipeline buffer sanitation utility.
Removes temporary CSV and image data from the processing pipeline.
Usage:
    python clear_buffer.py          # Execute removal
    python clear_buffer.py --dry-run # Preview only
"""
import os
import sys
import glob
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR
BUFFER_TARGETS = {
    "CSV buffer (processed)": {
        "dir": os.path.join(BASE_DIR, "data", "processed_csv"),
        "patterns": ["*.csv"],
    },
    "VLM dataset (vlm_dataset)": {
        "dir": os.path.join(BASE_DIR, "data", "vlm_dataset"),
        "patterns": ["chart/*.png","shap_img/*.png"],
    },
    "PCAP raw_pcap": {
        "dir": os.path.join(BASE_DIR, "data", "raw_pcap"),
        "patterns": ["*.pcap"],
    },
    "PCAP raw_data": {
        "dir": os.path.join(BASE_DIR, "data", "raw_data"),
        "patterns": ["*.pcap"],
    },
}
def collect_files(directory: str, patterns: list[str]) -> list[str]:
    """Return list of files matching specified patterns in directory."""
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(directory, pat)))
    return sorted(files)
def clear_buffer(dry_run: bool = False):
    """Clean all buffer files. dry_run mode only lists files without deleting."""
    mode_tag = "[DRY-RUN]" if dry_run else "[LIVE]"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[Buffer Cleanup] {mode_tag} Session - {now}")
    total_deleted = 0
    total_size = 0
    for label, cfg in BUFFER_TARGETS.items():
        directory = cfg["dir"]
        patterns  = cfg["patterns"]
        print(f"[Target] {label} -> {directory}")
        if not os.path.isdir(directory):
            print(f"[Warning] Directory does not exist: {directory}")
            continue
        files = collect_files(directory, patterns)
        if not files:
            print("[Info] No files to remove.")
            continue
        for fp in files:
            size = os.path.getsize(fp)
            total_size += size
            status = "[Preview]" if dry_run else "[Delete]"
            print(f"  {status} {os.path.basename(fp)} ({size:,} bytes)")
            if not dry_run:
                os.remove(fp)
                total_deleted += 1
        if dry_run:
            total_deleted += len(files)
    
    verb = "Proposed" if dry_run else "Total"
    print(f"\n[Summary] {verb} removal: {total_deleted} files ({total_size:,} bytes)")
    if dry_run:
        print("[Note] Re-run without --dry-run for actual deletion.")
if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    clear_buffer(dry_run=dry_run)
