"""Ghi ket qua va ghi lai phien ban code."""
import json
import os
import subprocess

import config


def append_jsonl(rec, path=None):
    """Ghi tung ban ghi ngay - mat ket noi giua chung khong mat du lieu da chay."""
    path = path or config.RESULTS_JSONL
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        # Thu muc ket qua read-only (vd LAYER2 nam o /kaggle/input): chuyen sang
        # /kaggle/working neu co, khong thi dung thu muc tam.
        import tempfile
        alt = "/kaggle/working" if os.path.isdir("/kaggle/working") else tempfile.gettempdir()
        path = os.path.join(alt, os.path.basename(path))
        print(f"  !! khong ghi duoc vao thu muc goc -> {path}")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def git_commit(*dirs):
    """Commit ngan cua repo code: thu lan luot cac thu muc (LAYER2 trong repo, hoac UTS_URA canh workspace)."""
    for d in dirs:
        try:
            c = subprocess.run(["git", "-C", str(d), "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True).stdout.strip()
            if c:
                return c
        except Exception:
            pass
    return None
