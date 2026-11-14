"""
Chang A cua Layer 2 cho MOT flow (CPU, khong goi VLM): dong flow tho -> bang chung day du de dung prompt.

    row (78 cot tho) -> Layer 1: p(x)                     chang "layer1"   (scale + AE + XGBoost)
                     -> p(x) >= tau_alpha ? (alert)        neu khong: dung, khong co bao cao
                     -> TreeSHAP phi (95 chieu)            chang "shap"
                     -> tra tap R: vote 5-NN, s(q), co      chang "lookup"   (retrieval/store.py)
                     -> bang C (chi khi bat co)             chang "contrast" (evidence/contrast.py)
                     -> chon doan KB                        chang "kb_select" (knowledge_base/retrieve.py)
                     -> 5 anh: SHAP waterfall + 4 chart     chang "images"

Ket qua la dict thuan JSON (pipeline.py ghi ra <sample_id>/evidence/). Thoi gian tung chang do bang
utils.timing.StageTimer (time.perf_counter), moi lan goi mot flow.
"""
from __future__ import annotations

import math
import os

import numpy as np

import config
from utils.timing import StageTimer

N_TOP = 5                       # so chieu |phi| lon nhat dua vao prompt ([E1]..[E5])
FACT_COLUMNS = ("Protocol", "Dst Port")   # luon dua vao prompt ([E6], [E7]) neu khong nam trong top


def _num(v):
    """Gia tri tho cua mot o: float neu doc duoc (NaN/Inf giu nguyen de KB tu bo qua), None neu khong phai so."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _json_num(v):
    return v if v is None or math.isfinite(v) else str(v)


class EvidenceBuilder:
    """Nap mot lan Layer 1, tap R va KB; build() chay cho tung flow."""

    def __init__(self, l1=None, store=None, kb=None, exclude_labels=None):
        from gate.engine import Layer1Engine
        from knowledge_base.retrieve import KnowledgeBase
        from retrieval.store import ReferenceStore

        self.l1 = l1 or Layer1Engine(config.VARIANT)
        self.exclude = list(config.RAG_EXCLUDE_LABELS if exclude_labels is None else exclude_labels)
        self.store = store or ReferenceStore.load(config.REFERENCE_SET_NPZ, exclude_labels=self.exclude)
        self.kb = kb or KnowledgeBase(strict=config.KB_STRICT)
        self.names = list(self.l1.feature_names)
        self.base = self.names[: self.l1.n_base]
        if self.store.feature_names != self.names:
            raise ValueError("Ten chieu cua tap R khac Layer 1: tap R phai build tu cung checkpoint")
        if self.store.x is None:
            raise ValueError("Tap R chua co gia tri tho (x__): chay retrieval/reference_set.py --add-raw")

    # ------------------------------------------------------------------------------------------
    def build(self, row, out_dir, timer=None):
        t = timer or StageTimer()
        x_raw = {c: _num(row[c]) for c in self.base}
        X = np.array([[np.nan if x_raw[c] is None else x_raw[c] for c in self.base]], dtype="float64")

        with t("layer1"):
            A = self.l1.build_input(X)
            p = float(self.l1.model.predict_proba(A)[0, 1])
        rec = {"p": p, "tau_alpha": float(self.l1.threshold), "alert": bool(p >= self.l1.threshold)}
        if not rec["alert"]:
            return rec

        with t("shap"):
            phi = np.asarray(self.l1.shap_values(A)[0], dtype="float64")
        with t("lookup"):
            look = self.store.lookup(phi)

        top = self._top_features(phi, A[0], x_raw)
        contrast = None
        if look.flagged:
            from evidence.contrast import contrast_table
            with t("contrast"):
                contrast = contrast_table(phi, x_raw, self.store, look.winners, look.label,
                                          self.kb.index["discrete_features"])
        with t("kb_select"):
            sel = self.kb.select(look.label, look.flagged, [f["name"] for f in top], x_raw,
                                 contrast_features=[r["feature_key"] for r in contrast["rows"]] if contrast else (),
                                 exclude_labels=self.exclude)
        with t("images"):
            images = self._images(row, phi, A[0], p, out_dir)

        rec.update({
            "top_features": top,
            "x_facts": self._facts(x_raw, top),
            "lookup": {
                "label": look.label, "votes": look.votes, "novelty": look.novelty, "tau_zd": look.tau_zd,
                "flagged": look.flagged, "n_winners": int(len(look.winners)),
                "neighbours": [{"label": str(self.store.labels[i]), "distance": float(d)}
                               for i, d in zip(look.neighbours, look.distances)],
            },
            "contrast": contrast,
            "kb": {"sha256": sel.kb_sha256, "techniques": sel.techniques, "passages": sel.passages,
                   "mitigations": sel.mitigations, "missing": sel.missing},
            "images": images,
        })
        return rec

    # ------------------------------------------------------------------------------------------
    def _top_features(self, phi, a, x_raw):
        """N_TOP chieu |phi| lon nhat: [E1]..[E5] voi phi, gia tri model thay (da scale) va gia tri tho."""
        idx = np.argsort(-np.abs(phi))[:N_TOP]
        return [{"id": f"E{k}", "name": self.names[i], "phi": float(phi[i]), "model_value": float(a[i]),
                 "raw_value": _json_num(x_raw[self.names[i]]) if self.names[i] in x_raw else None}
                for k, i in enumerate(idx, 1)]

    @staticmethod
    def _facts(x_raw, top):
        """Protocol va Dst Port luon co trong prompt ([E6], [E7]) neu chua nam trong top."""
        have = {f["name"] for f in top}
        out, k = [], len(top)
        for c in FACT_COLUMNS:
            if c not in have:
                k += 1
                out.append({"id": f"E{k}", "name": c, "raw_value": _json_num(x_raw.get(c))})
        return out

    def _images(self, row, phi, a, p, out_dir):
        """5 anh: SHAP waterfall (gia tri model thay, khop SHAP) + 4 bieu do luong tu gia tri tho."""
        from utils import plot_flow, plot_shap
        os.makedirs(out_dir, exist_ok=True)
        shap_png = plot_shap.save_waterfall(phi, a, self.names, p, os.path.join(out_dir, "shap_waterfall.png"))
        return [shap_png] + list(plot_flow.generate_flow_charts(row, out_dir))
