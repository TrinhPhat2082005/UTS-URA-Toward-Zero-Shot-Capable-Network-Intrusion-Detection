"""
Luc chay: chon doan KB cho MOT alert. Khong dung yaml hay mang; chi doc store/passages.jsonl + index.json.

    from knowledge_base.retrieve import KnowledgeBase
    kb = KnowledgeBase()
    sel = kb.select(label="DoS_attacks_Hulk", flagged=False,
                    top_features=["Fwd Seg Size Min", "Dst Port", ...],   # 5 chieu |phi| lon nhat
                    x_raw={"Protocol": 6, "Dst Port": 80, ...})           # gia tri tho cua flow

Luat chon:
  M, B    theo nhan l* (vote 5-NN) -> technique theo Table II cua Borisenko et al. 2022 (index.json):
            M-<nhan>             cau mapping nhan -> tactic / technique / sub-technique
            moi technique t:     B-<t> (mo ta ATT&CK), B-DET-<t> (analytic mang, neu co), B-<CAPEC> lien ket
          Khong co co: role "attack_type_background" (mo ta LOAI tan cong cua nhan tien le, khong mo ta flow nay).
          Co co (zero-day): role "comparison_only"; bo M (doan mapping chi ra technique); B giu noi dung nhung an
          tieu de va ma trich dan (cite = REF-MECH-n), vi report khong duoc gan technique khi bat co.
          Protocol A: nhan trong exclude_labels khong bao gio duoc chon.
  Moi doan tra them `cite`: ma de VLM trich dan (= pid, tru B khi co co).
  H1      dinh nghia moi feature trong top_features (+ cac dong cua bang doi chieu C khi co co);
          16 chieu ae_latent gop thanh mot doan; H1-DIRECTION khi co feature Fwd/Bwd.
  H2      giao thuc va cong cua alert (moi cong co dang ky IANA); co doan cong thi kem H2-PORT-NOTE (RFC 6335);
          co TCP: doan cua co MA COT THUC SU CHUA (index.json flag_column_holds), khong theo ten cot.
  H3      moi doan co `when` dung voi x_raw.
  G       ung vien mitigation cua moi technique (ATT&CK mitigates + CAPEC Mitigations), role "mitigation_option",
          dua VLM de chon cho muc Mitigations (trich [K:G-...]); sel.mitigations giu cung danh sach de cham.
          Khong co khi bat co.

Xu ly loi:
  - Protocol / Dst Port khong huu han (NaN, Inf, chuoi) -> bo qua doan H2 cua truong do; dieu kien `when`
    voi gia tri khong huu han -> khong dung (conditions.py).
  - pid can chon ma KB khong co -> ghi vao sel.missing. Thieu H2-PORT (cong khong dang ky IANA) / H2-PROTO
    (giao thuc 0) la binh thuong nen khong canh bao; thieu M/B/H1/H3 -> warnings.warn, hoac raise khi strict=True.
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
EXPECTED_MISSING = ("H2-PORT-", "H2-PROTO-")


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


@dataclass
class Selection:
    passages: list = field(default_factory=list)      # [{pid, cite, part, role, reason, title, text}] theo thu tu
    mitigations: list = field(default_factory=list)   # [{pid, title, text}] ung vien G (cung co trong passages)
    missing: list = field(default_factory=list)       # [{pid, reason}] pid can chon nhung KB khong co
    techniques: list = field(default_factory=list)    # technique cua nhan (rong khi bat co / nhan bi loai)
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
        self.mitigations_of = {}
        for p in self.passages.values():
            if p["part"] == "G":
                self.mitigations_of.setdefault(p["keys"]["technique"], []).append(p)

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _text(p):
        """Text dua VLM: doan + cac ghi chu (moi phan ket thuc bang dau cau)."""
        parts = [p["text"]] + [n["text"] for n in p.get("notes", [])]
        return " ".join(s if s.rstrip().endswith((".", ")", "?", "!")) else s.rstrip() + "." for s in parts)

    def _add(self, sel, seen, pid, role, reason, flagged=False):
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
        title, cite = p.get("title"), pid
        if flagged and p["part"] == "B":
            # tieu de va pid cua B chua ID/ten technique hoac CAPEC
            n = sum(1 for q in sel.passages if q["part"] == "B") + 1
            title, cite = "Attack mechanism of the nearest reference label", f"REF-MECH-{n}"
        sel.passages.append({"pid": pid, "cite": cite, "part": p["part"], "role": role, "reason": reason,
                             "title": title, "text": self._text(p)})
        return True

    @staticmethod
    def _h1_pid(feature):
        # "ae_latent_3" (top feature) va "ae_latent" (feature_key dong gop cua bang C) -> mot doan
        return "H1-ae_latent" if feature.startswith("ae_latent") else f"H1-{pid_safe(feature)}"

    # ------------------------------------------------------------------------------------------
    def select(self, label, flagged, top_features, x_raw, contrast_features=(), exclude_labels=()):
        sel, seen = Selection(kb_sha256=self.index.get("kb_sha256", "")), set()
        info = self.index["labels"].get(label)
        if info is None and label is not None:
            sel.missing.append({"pid": f"M-{label}", "reason": "label not in Borisenko Table II"})
        use_label = info is not None and label not in set(exclude_labels)
        role = "comparison_only" if flagged else "attack_type_background"

        # M, B theo nhan -> technique (Borisenko 2022, Table II)
        if use_label:
            if not flagged:
                sel.techniques = list(info["techniques"])
                self._add(sel, seen, f"M-{label}", role, f"ATT&CK mapping of reference label {label}")
            for tid in info["techniques"]:
                why = f"technique of reference label {label} (Borisenko et al. 2022, Table II)"
                self._add(sel, seen, f"B-{tid}", role, why, flagged)
                if info["detection"].get(tid):
                    self._add(sel, seen, f"B-DET-{tid}", role, f"network detection analytics of {why}", flagged)
                for cid in info["capec"].get(tid, []):
                    self._add(sel, seen, f"B-{cid}", role, f"CAPEC pattern linked to {why}", flagged)

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
        if port is not None and self._add(sel, seen, f"H2-PORT-{port}", "definition", f"Dst Port = {port}"):
            self._add(sel, seen, "H2-PORT-NOTE", "caveat", "a port passage was selected")
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

        # G: ung vien mitigation (ATT&CK + CAPEC) cua moi technique, dua VLM de chon cho muc Mitigations;
        #    sel.mitigations giu danh sach ung vien de cham (muc VLM chon phai thuoc danh sach). Khong co khi bat co.
        if use_label and not flagged:
            for tid in info["techniques"]:
                for p in self.mitigations_of.get(tid, []):
                    self._add(sel, seen, p["pid"], "mitigation_option",
                              f"mitigation candidate for technique {tid} of reference label {label}")
            sel.mitigations = [{"pid": p["pid"], "title": p["title"], "text": p["text"]}
                               for tid in info["techniques"] for p in self.mitigations_of.get(tid, [])]
        return sel
