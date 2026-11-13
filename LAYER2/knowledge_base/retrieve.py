"""
Luc chay: chon doan KB cho MOT alert. Khong dung yaml hay mang; chi doc store/passages.jsonl + index.json.

    from knowledge_base.retrieve import KnowledgeBase
    kb = KnowledgeBase()
    sel = kb.select(label="DoS_attacks_Hulk", flagged=False,
                    top_features=["Fwd Seg Size Min", "Dst Port", ...],   # 5 chieu |phi| lon nhat
                    x_raw={"Protocol": 6, "Dst Port": 80, ...})           # gia tri tho cua flow

Luat chon (plan_kb.md muc 10, da chot):
  A', B'  theo nhan l* (vote 5-NN). Khong co co: role "alert_label". Co co: role "comparison_only".
          Protocol A: nhan trong exclude_labels khong bao gio duoc chon.
  H1      dinh nghia moi feature trong top_features (+ cac dong cua bang doi chieu C khi co co);
          16 chieu ae_latent gop thanh mot doan; H1-DIRECTION khi co feature Fwd/Bwd.
  H2      protocol va cong cua alert (neu KB co doan); co TCP: doan cua co MA COT THUC SU CHUA.
  H3      moi doan co `when` dung voi x_raw.
  G'      mitigation cua technique, tra rieng cho code dien muc 6 (khong dua VLM); khong co khi bat co.

Xu ly loi (checklist 2.1, 2.2):
  - Protocol / Dst Port khong huu han (NaN, Inf, chuoi) -> bo qua doan H2 cua truong do; dieu kien `when`
    voi gia tri khong huu han -> khong dung (conditions.py).
  - pid can chon ma KB khong co -> ghi vao sel.missing. Thieu H2-PORT/H2-PROTO la binh thuong (KB chi co
    cong chinh) nen khong canh bao; thieu A'/B'/H1/H3 -> warnings.warn, hoac raise khi strict=True.
"""
from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from knowledge_base import conditions

STORE = Path(__file__).resolve().parent / "store"


def pid_safe(name):
    name = name.replace("/", "_per_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _finite_int(v):
    """int(v) neu v la so huu han, nguoc lai None (flow live co the co NaN/Inf)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if math.isfinite(f) else None


EXPECTED_MISSING = ("H2-PORT-", "H2-PROTO-")   # KB chi co doan cho cong/giao thuc chinh


@dataclass
class Selection:
    passages: list = field(default_factory=list)      # [{pid, part, role, reason, title, text}] theo thu tu dua VLM
    mitigations: list = field(default_factory=list)   # [{pid, title, text}] cho code dien muc 6
    missing: list = field(default_factory=list)       # [{pid, reason}] pid can chon nhung KB khong co
    kb_sha256: str = ""

    def pids(self):
        return [p["pid"] for p in self.passages]


class KnowledgeBase:
    def __init__(self, store_dir=STORE, strict=False):
        store_dir = Path(store_dir)
        self.strict = strict
        self.passages = {}
        for line in (store_dir / "passages.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = json.loads(line)
                self.passages[p["pid"]] = p
        self.index = json.loads((store_dir / "index.json").read_text(encoding="utf-8"))
        self.h3 = [p for p in self.passages.values() if p["part"] == "H3"]

    # ------------------------------------------------------------------------------------------
    def _text(self, p):
        """Text dua VLM: doan + cac ghi chu cua no (moi phan ket thuc bang dau cau)."""
        parts = [p["text"]] + [n["text"] for n in p.get("notes", [])]
        return " ".join(s if s.rstrip().endswith((".", ")", "?", "!")) else s.rstrip() + "." for s in parts)

    def _add(self, sel, seen, pid, role, reason):
        if pid in seen:
            return False
        if pid not in self.passages:
            sel.missing.append({"pid": pid, "reason": reason})
            if not pid.startswith(EXPECTED_MISSING):
                msg = f"KB khong co doan {pid} ({reason})"
                if self.strict:
                    raise KeyError(msg)
                warnings.warn(msg, stacklevel=3)
            return False
        p = self.passages[pid]
        seen.add(pid)
        sel.passages.append({"pid": pid, "part": p["part"], "role": role, "reason": reason,
                             "title": p.get("title"), "text": self._text(p)})
        return True

    def _h1_pid(self, feature):
        # "ae_latent_3" (top feature) va "ae_latent" (feature_key dong gop cua bang C) -> mot doan
        return "H1-ae_latent" if feature.startswith("ae_latent") else f"H1-{pid_safe(feature)}"

    # ------------------------------------------------------------------------------------------
    def select(self, label, flagged, top_features, x_raw, contrast_features=(), exclude_labels=()):
        sel, seen = Selection(kb_sha256=self.index.get("kb_sha256", "")), set()
        info = self.index["labels"].get(label)
        role = "comparison_only" if flagged else "alert_label"

        # A', B' theo nhan
        if info and label not in set(exclude_labels):
            self._add(sel, seen, f"A-{label}", role, f"label from 5-NN vote: {label}")
            self._add(sel, seen, f"B-{info['technique']}", role, f"technique of {label}")
            if info.get("capec"):
                self._add(sel, seen, f"B-{info['capec']}", role, f"CAPEC entry of {label}")

        # H1: top feature, roi cac dong cua bang C (chi khi co co)
        feats = list(top_features) + (list(contrast_features) if flagged else [])
        for i, f in enumerate(feats):
            why = "top-|phi| feature" if i < len(top_features) else "row of contrast table C"
            self._add(sel, seen, self._h1_pid(f), "definition", why)
        if any(("Fwd" in f or "Bwd" in f) for f in feats):
            self._add(sel, seen, "H1-DIRECTION", "definition", "a selected feature is directional")

        # H2: giao thuc, cong, co
        proto, port = _finite_int(x_raw.get("Protocol")), _finite_int(x_raw.get("Dst Port"))
        if proto is not None:
            self._add(sel, seen, f"H2-PROTO-{proto}", "definition", f"Protocol = {proto}")
        if port is not None:
            self._add(sel, seen, f"H2-PORT-{port}", "definition", f"Dst Port = {port}")
        holds = self.index["flag_column_holds"]
        for f in feats:
            if f in holds:
                flag = holds[f]
                self._add(sel, seen, f"H2-FLAG-{flag}", "definition", f"column '{f}' holds the {flag} flag")
                if flag in ("ECE", "CWR"):
                    self._add(sel, seen, "H2-FLAG-ECN-SETUP", "definition", f"column '{f}' holds the {flag} flag")

        # H3: artefact co dieu kien
        for p in self.h3:
            if conditions.holds(p["when"], x_raw):
                self._add(sel, seen, p["pid"], "caveat", "condition holds on this flow")

        # G': mitigation cho code (khong dua VLM); khong co khi bat co
        if info and not flagged:
            pre = f"G-{info['technique']}-"
            sel.mitigations = [{"pid": p["pid"], "title": p["title"], "text": p["text"]}
                               for pid, p in sorted(self.passages.items()) if pid.startswith(pre)]
        return sel
