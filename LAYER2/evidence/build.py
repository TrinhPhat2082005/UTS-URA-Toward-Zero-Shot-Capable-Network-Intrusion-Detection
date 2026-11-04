"""
Chang A cua Layer 2 (CPU): SHAP vector cua alert -> ket qua retriever + context cho VLM + 5 anh.

Ket qua cua chang nay (evidence) la tat ca nhung gi VLM can; chang B (vlm/runner.py + prompts.py)
khong can Layer 1 hay tap R. Pipeline live chay hai chang lien nhau (pipeline.analyze_sample);
Setup 1/2 ghi evidence thanh bundle (evidence/bundle.py) roi chay VLM tren Kaggle.
"""
import os
import time

import numpy as np

import config
from knowledge_base.profiles import LABEL_TO_PROFILE_ID


class EvidenceBuilder:
    """Sinh RAG context va 5 anh cho MOT flow, ngay tai thoi diem can."""

    def __init__(self, retriever=None):
        if retriever is None:
            from retrieval.retriever import SHAPKNNRetriever

            retriever = SHAPKNNRetriever(
                npz_path=config.REFERENCE_SET_NPZ,
                profiles_dir=config.ATTACK_PROFILES_DIR,
                tau_zd=config.RAG_ZERODAY_THRESHOLD,
                k=config.RAG_NOVELTY_K,
                exclude_labels=config.RAG_EXCLUDE_LABELS,
            )
        self.retriever = retriever

    def retrieve(self, shap_vector):
        """SHAP vector -> ket qua k-NN + khoi text cho VLM."""
        import prompts

        r = self.retriever
        t0 = time.perf_counter()
        res = r.retrieve_batch([shap_vector], top_k=config.RAG_TOP_K)[0]
        ctx = prompts.format_retrieval_context(res, r.k, r.tau_zd, r.unknown_profile)
        return res, ctx, time.perf_counter() - t0

    def make_images(self, row, shap_vector, feature_names, out_dir, proba,
                    model_input=None):
        """Ve 1 SHAP waterfall + 4 bieu do luu luong. Tra danh sach duong dan.

        Args:
            feature_names: ten cua TAT CA chieu trong shap_vector (95 voi AE),
                khong phai 78 cot goc.
            model_input: vector da scale + augmented ma model thuc su nhin thay.
                Waterfall hien gia tri nay, khong phai gia tri tho, de khop voi
                SHAP dang giai thich.
        """
        from utils import plot_flow, plot_shap

        os.makedirs(out_dir, exist_ok=True)
        t0 = time.perf_counter()

        shap_vector = np.asarray(shap_vector, dtype="float64").ravel()
        if model_input is None:
            model_input = row[list(feature_names)[:len(shap_vector)]].to_numpy("float64")
        model_input = np.asarray(model_input, dtype="float64").ravel()

        if not (len(shap_vector) == len(model_input) == len(feature_names)):
            raise ValueError(
                f"Lech so chieu: shap={len(shap_vector)}, "
                f"data={len(model_input)}, names={len(feature_names)}"
            )

        shap_png = plot_shap.save_waterfall(shap_vector, model_input, feature_names, proba,
                                            os.path.join(out_dir, "shap_waterfall.png"))
        chart_paths = plot_flow.generate_flow_charts(row, out_dir)
        return [shap_png] + list(chart_paths), time.perf_counter() - t0


def candidates(labels, distances):
    """3 ung vien cua retriever theo dinh dang manifest: nhan, khoang cach, id profile; gan nhat truoc."""
    return [{"label": str(lab), "distance": float(d), "profile_id": LABEL_TO_PROFILE_ID[str(lab)]}
            for lab, d in zip(labels, distances)]


def top_features(shap_vector, feature_names, n=5):
    """n chieu dong gop lon nhat (gia tri tuyet doi) cua vector SHAP."""
    top = np.argsort(-np.abs(shap_vector))[:n]
    return [{"name": feature_names[k], "value": float(shap_vector[k])} for k in top]
