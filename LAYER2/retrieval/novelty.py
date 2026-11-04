"""Co zero-day cua retriever (paper Eq. zeroday-flag).

    s(q)    = trung binh L2 tu q toi k = 10 vector gan nhat cua R      (SHAPKNNRetriever.score)
    co      = s(q) > tau_zd
    tau_zd  = Q_0.95 cua s tren alert da biet giu ngoai R               (experiments/rq3_retrieval/02_calibrate_novelty.py)
              luu trong <tap R>.meta.json, khoa "novelty" -> "tau_zd"

Co luon duoc tinh va ghi vao evidence. Co dua lua chon UNKNOWN cho VLM hay khong la viec cua prompt
(prompts.build_candidate_prompt, tham so allow_unknown: Setup 1 False, Setup 2 va pipeline True),
nen bat/tat UNKNOWN khong can build lai evidence.
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
