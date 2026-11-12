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
  required_passage_ids = H1 cua top_features + moi H3 duoc chon (dung de cham context recall).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from knowledge_base import conditions

STORE = Path(__file__).resolve().parent / "store"


def pid_safe(name):
    name = name.replace("/", "_per_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


@dataclass
class Selection:
    passages: list = field(default_factory=list)      # [{pid, part, role, reason, title, text}] theo thu tu dua VLM
    required: list = field(default_factory=list)      # pid bat buoc phai duoc trich (cham do phu)
    mitigations: list = field(default_factory=list)   # [{pid, title, text}] cho code dien muc 6
    kb_sha256: str = ""

    def pids(self):
        return [p["pid"] for p in self.passages]


class KnowledgeBase:
    def __init__(self, store_dir=STORE):
        store_dir = Path(store_dir)
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
        if pid in seen or pid not in self.passages:
            return False
        p = self.passages[pid]
        seen.add(pid)
        sel.passages.append({"pid": pid, "part": p["part"], "role": role, "reason": reason,
                             "title": p.get("title"), "text": self._text(p)})
        return True

    def _h1_pid(self, feature):
        return "H1-ae_latent" if feature.startswith("ae_latent_") else f"H1-{pid_safe(feature)}"

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
            pid = self._h1_pid(f)
            self._add(sel, seen, pid, "definition", why)
            if i < len(top_features) and pid not in sel.required:
                sel.required.append(pid)
        if any(("Fwd" in f or "Bwd" in f) for f in feats):
            self._add(sel, seen, "H1-DIRECTION", "definition", "a selected feature is directional")

        # H2: giao thuc, cong, co
        proto, port = x_raw.get("Protocol"), x_raw.get("Dst Port")
        if proto is not None:
            self._add(sel, seen, f"H2-PROTO-{int(proto)}", "definition", f"Protocol = {int(proto)}")
        if port is not None:
            self._add(sel, seen, f"H2-PORT-{int(port)}", "definition", f"Dst Port = {int(port)}")
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
                if self._add(sel, seen, p["pid"], "caveat", "condition holds on this flow"):
                    sel.required.append(p["pid"])

        # G': mitigation cho code (khong dua VLM); khong co khi bat co
        if info and not flagged:
            pre = f"G-{info['technique']}-"
            sel.mitigations = [{"pid": p["pid"], "title": p["title"], "text": p["text"]}
                               for pid, p in sorted(self.passages.items()) if pid.startswith(pre)]
        return sel
