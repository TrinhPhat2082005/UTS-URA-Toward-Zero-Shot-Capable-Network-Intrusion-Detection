"""
Danh gia dieu kien `when` cua doan H3 tren gia tri tho cua mot flow, KHONG dung eval().

    when = {"all": [["TotLen Fwd Pkts", "==", 0], ["TotLen Bwd Pkts", "==", 0]]}
    when = {"any": [["Init Fwd Win Byts", "==", -1], ["Init Bwd Win Byts", "==", -1]]}

Dung chung cho luc build (dem tan suat) va luc chay (retrieve.py chon doan cho alert).
"""
from __future__ import annotations

import math
import operator

OPS = {"==": operator.eq, "!=": operator.ne, ">": operator.gt, ">=": operator.ge,
       "<": operator.lt, "<=": operator.le}


def _term(x_raw, col, op, value):
    v = x_raw.get(col)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False                      # flow live co the thieu cot / gia tri khong phai so
    if not math.isfinite(f):
        return False                      # NaN / Inf (CICFlowMeter co the sinh Inf khi chia 0): dieu kien khong dung
    return OPS[op](f, float(value))


def holds(when, x_raw):
    """True neu dieu kien dung voi flow. when=None -> True (doan khong co dieu kien)."""
    if not when:
        return True
    if "all" in when:
        return all(_term(x_raw, *t) for t in when["all"])
    if "any" in when:
        return any(_term(x_raw, *t) for t in when["any"])
    raise ValueError(f"dieu kien khong hop le: {when}")


def columns(when):
    """Cac cot ma dieu kien dung toi (de kiem ten cot luc build)."""
    return [t[0] for t in (when or {}).get("all", []) + (when or {}).get("any", [])]
