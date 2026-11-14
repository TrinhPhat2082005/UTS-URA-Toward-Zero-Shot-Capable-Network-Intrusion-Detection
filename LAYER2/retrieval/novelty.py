"""Co "khong co tien le gan" cua buoc tra tap R (paper Eq. zeroday-flag).

    s(q)    = trung binh L2 tu q toi k = 10 vector gan nhat cua R      (retrieval/store.py, ReferenceStore.lookup)
    co      = s(q) > tau_zd
    tau_zd  = Q_0.95 cua s tren alert da biet giu ngoai R               (experiments/rq3_retrieval/02_calibrate_novelty.py)
              luu trong <tap R>.meta.json, khoa "novelty" -> "tau_zd"

Co luon duoc tinh va ghi vao evidence. Khi co bat, prompts.py chuyen sang nhanh nearest_background (khong gan
technique, bang C, danh gia variant / different / undetermined).
"""
import json
from pathlib import Path

import numpy as np

QUANTILE = 0.95


def meta_path(npz_path):
    return Path(npz_path).with_suffix(".meta.json")


def read_tau(npz_path):
    """tau_zd da hieu chinh cua tap R, hoac None neu chua hieu chinh (khi do co khong bao gio bat)."""
    p = meta_path(npz_path)
    if not p.exists():
        return None
    return (json.loads(p.read_text(encoding="utf-8")).get("novelty") or {}).get("tau_zd")


def calibrate_tau(scores, quantile=QUANTILE):
    """Nguong tau_zd tu diem novelty cua cac alert da biet."""
    return float(np.quantile(scores, quantile))


def is_flagged(score, tau_zd):
    return bool(tau_zd is not None and score > tau_zd)
