"""
Buoc 1 cua KB: tai moi nguon trong sources.yaml ve $UTS_DATA/kb_sources/<id>/ va khoa bang sha256.

    python LAYER2/knowledge_base/build/fetch_sources.py            # tai nguon con thieu, ghi lock
    python LAYER2/knowledge_base/build/fetch_sources.py --verify   # khong tai; so sha256 voi lock

Lock (store/sources.lock.json) duoc commit; file nguon thi khong (dung luong, giay phep). Nguoi khac
tai lai bang script nay va --verify cho biet file co dung la file da dung de build KB khong.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import urllib.request

import common as C

UA = "Mozilla/5.0 (research knowledge-base builder)"


def _ssl_context():
    """Kho chung chi cua Windows trong ai_env bi loi (ASN1 NOT_ENOUGH_DATA): dung certifi neu co."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as r, open(dest, "wb") as f:
        f.write(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="khong tai, chi so sha256 voi lock")
    ap.add_argument("--refresh", action="store_true", help="tai lai ca file da co")
    args = ap.parse_args()

    sources, lock = C.load_sources(), C.load_lock()
    today = dt.date.today().isoformat()
    bad = 0
    for sid, spec in sources.items():
        entry = lock.get(sid, {"version": spec["version"], "files": {}})
        for f in spec.get("files", []):
            path = C.source_path(sid, f["name"])
            if args.verify:
                ok = path.exists() and C.sha256_file(path) == entry["files"].get(f["name"], {}).get("sha256")
                bad += not ok
                print(f"  [{'ok ' if ok else 'BAD'}] {sid}/{f['name']}")
                continue
            if args.refresh or not path.exists():
                print(f"  tai {sid}/{f['name']} ...", flush=True)
                download(f["url"], path)
                entry["retrieved"] = today
            entry["files"][f["name"]] = {"url": f["url"], "sha256": C.sha256_file(path),
                                         "bytes": path.stat().st_size}
        for f in spec.get("local", []):
            path = C.source_path(sid, f["name"])
            sha = C.sha256_file(path)
            if args.verify:
                ok = sha == entry["files"].get(f["name"], {}).get("sha256")
                bad += not ok
                print(f"  [{'ok ' if ok else 'CHANGED'}] {sid}/{f['name']} (file trong repo)")
                continue
            entry["files"][f["name"]] = {"path": f["path"], "sha256": sha, "bytes": path.stat().st_size}
            entry["retrieved"] = today
        entry["version"] = spec["version"]
        lock[sid] = entry
    if args.verify:
        print(f"\n{bad} file khong khop lock" if bad else "\nmoi file khop lock")
        sys.exit(1 if bad else 0)
    C.write_json(C.LOCK_JSON, lock)
    print(f"-> {C.LOCK_JSON} ({len(lock)} nguon); file nguon o {C.SOURCES_DIR}")


if __name__ == "__main__":
    main()
