"""
Bundle evidence: cau noi giua chang A (CPU, may local) va chang B (VLM, GPU Kaggle).

    <bundle>/manifest.jsonl      mot dong / alert: nhan that, Layer 1, novelty, co, 3 ung vien,
                                 top feature SHAP, duong dan anh (tuong doi so voi bundle)
    <bundle>/phi.npy             vector SHAP day du cua cac alert, cung thu tu voi manifest
    <bundle>/bundle_meta.json    cau hinh, phien ban code, thoi diem tao (+ file json khac, vd fold_stats.json)
    <bundle>/images/<alert_id>/  5 anh (SHAP waterfall + 4 bieu do luong)

Chang B chi can bundle + prompts.py + knowledge_base/ + vlm/runner.py: doi prompt hay bat/tat
UNKNOWN (allow_unknown) thi chay lai tren cung bundle, khong build lai evidence.
"""
import json
from pathlib import Path

import numpy as np


def write(out_dir, manifest, meta, phi=None, extra=None):
    """Ghi manifest.jsonl, bundle_meta.json, phi.npy (neu co) va cac file json trong extra."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if phi is not None:
        assert len(phi) == len(manifest), "phi va manifest phai cung so dong"
        np.save(out / "phi.npy", np.asarray(phi, dtype="float32"))
    for name, obj in (extra or {}).items():
        (out / name).write_text(json.dumps(obj, indent=1), encoding="utf-8")
    (out / "bundle_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def read(bundle_dir):
    """(manifest, bundle_meta, phi hoac None)."""
    d = Path(bundle_dir)
    manifest = [json.loads(line) for line in (d / "manifest.jsonl").open(encoding="utf-8") if line.strip()]
    meta = json.loads((d / "bundle_meta.json").read_text(encoding="utf-8"))
    phi = np.load(d / "phi.npy") if (d / "phi.npy").exists() else None
    return manifest, meta, phi


def rel(path, bundle_dir):
    """Duong dan anh ghi trong manifest: tuong doi so voi bundle, dau '/'."""
    return str(Path(path).relative_to(bundle_dir)).replace("\\", "/")
