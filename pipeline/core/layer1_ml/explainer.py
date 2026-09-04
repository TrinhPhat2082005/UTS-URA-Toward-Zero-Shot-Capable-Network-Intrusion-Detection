import shap
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
CLASS_NAMES = [
    "Benign",
    "Anomaly/Malicious"
]

class IDSExplainer:
    def __init__(self, model_booster, feature_names):
        """Initialize SHAP explainer for XGBoost model."""
        self.model = model_booster
        self.feature_names = feature_names
        self.output_path = "project_root/data/vlm_dataset/shap_img"
        os.makedirs(self.output_path, exist_ok=True)
        self.explainer = shap.TreeExplainer(self.model)

    def _compute_shap(self, ids_df):
        """Run TreeExplainer once and return the Explanation object.

        All public methods call this so TreeExplainer is never run twice
        on the same DataFrame within a single pipeline cycle.
        """
        X = ids_df[self.feature_names].astype('float32')
        return self.explainer(X)

    def _extract_values(self, shap_exp, i, label):
        """Extract per-class SHAP values for sample i."""
        if shap_exp.values.ndim == 3:
            return shap_exp.values[i, :, label]
        return shap_exp.values[i]

    def explain_and_get_dicts(self, ids_df, predictions, top_k=5):
        """Run TreeExplainer once, return waterfall PNGs + per-flow SHAP dicts
        + full per-flow SHAP vectors.

        This is the primary entry point for the pipeline (replaces calling
        explain_samples + get_shap_dicts_batch separately). full_vectors is
        included so callers needing the complete feature vector (e.g.
        SHAPCentroidRetriever's cosine matching against centroids) don't have
        to re-run TreeExplainer a second time on the same flows.

        Returns:
            image_paths:  list[str] — one waterfall PNG per anomalous flow
            shap_dicts:   list[dict] — {feature: shap_value} top-k only (for RAG text/logging)
            full_vectors: list[np.ndarray] — full-length SHAP vector per flow, in
                          self.feature_names order (for centroid cosine matching)
        """
        if ids_df.empty:
            return [], [], []

        shap_exp = self._compute_shap(ids_df)
        image_paths = []
        shap_dicts = []
        full_vectors = []

        for i in range(len(ids_df)):
            label = int(predictions[i])
            values = self._extract_values(shap_exp, i, label)
            full_vectors.append(values)

            # --- waterfall PNG ---
            peer_ip = ids_df.iloc[i].get('Peer_IP', 'Unknown')
            if shap_exp.values.ndim == 3:
                base_val = (
                    float(shap_exp.base_values[i, label])
                    if shap_exp.base_values.ndim == 2
                    else float(shap_exp.base_values[i])
                )
                explanation = shap.Explanation(
                    values=values,
                    base_values=base_val,
                    data=shap_exp.data[i],
                    feature_names=self.feature_names
                )
            else:
                explanation = shap_exp[i]

            class_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label)
            shap.plots.waterfall(explanation, max_display=10, show=False)
            plt.title(f"SHAP Analysis | Peer: {peer_ip} | {class_name} (id={label})")
            img_name = f"SHAP_img_{i+1}.png"
            full_path = os.path.join(self.output_path, img_name)
            plt.savefig(full_path, bbox_inches='tight')
            plt.close()
            image_paths.append(full_path)
            print(f"[XAI] Generated: {img_name}")

            # --- SHAP dict for RAG ---
            top_indices = np.argsort(np.abs(values))[-top_k:][::-1]
            shap_dicts.append({self.feature_names[idx]: float(values[idx]) for idx in top_indices})

        return image_paths, shap_dicts, full_vectors



    def explain_samples(self, ids_df, predictions):
        """Generate SHAP waterfall plots for each sample."""
        image_paths, _, _ = self.explain_and_get_dicts(ids_df, predictions)
        return image_paths

    def get_shap_dicts_batch(self, ids_df, predictions, top_k=5):
        """Return a list of SHAP dicts for all samples."""
        _, shap_dicts, _ = self.explain_and_get_dicts(ids_df, predictions, top_k=top_k)
        return shap_dicts

    def get_shap_dict(self, ids_df, predictions, sample_idx=0, top_k=5):
        """Return top-K SHAP features for a single sample."""
        _, shap_dicts, _ = self.explain_and_get_dicts(ids_df, predictions, top_k=top_k)
        return shap_dicts[sample_idx]

    def get_top_features_str(self, ids_df, predictions, top_k=3):
        """Extract top K important features as a formatted string."""
        _, shap_dicts, _ = self.explain_and_get_dicts(ids_df, predictions, top_k=top_k)
        reports = []
        for i, (d, pred) in enumerate(zip(shap_dicts, predictions)):
            label = int(pred)
            class_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label)
            top_feats = [f"{feat} ({val:.2f})" for feat, val in d.items()]
            reports.append(f"Flow {i+1} [{class_name}] primary drivers: {', '.join(top_feats)}")
        return "\n".join(reports)
