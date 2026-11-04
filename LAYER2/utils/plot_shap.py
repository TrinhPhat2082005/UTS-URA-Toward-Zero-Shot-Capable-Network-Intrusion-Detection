"""Anh SHAP waterfall cua mot alert (anh 1 trong 5 anh gui VLM; 4 anh con lai o plot_flow.py)."""


def save_waterfall(shap_vector, model_input, feature_names, proba, out_png):
    """Ve waterfall 12 dong dong gop lon nhat va luu ra out_png.

    Args:
        shap_vector: vector SHAP day du (95 chieu voi AE).
        model_input: vector da scale + augmented ma model thuc su nhin thay; waterfall hien
            gia tri nay, khong phai gia tri tho, de khop voi SHAP dang giai thich.
        feature_names: ten cua TAT CA chieu trong shap_vector.
        proba: p(x) cua Layer 1, in tren tieu de.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    expl = shap.Explanation(
        values=shap_vector,
        base_values=0.0,
        data=model_input,
        feature_names=list(feature_names),
    )
    plt.figure(figsize=(7.5, 5))
    shap.plots.waterfall(expl, max_display=12, show=False)
    plt.title(f"Layer 1 attribution (p_attack = {proba:.3f})", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close("all")
    return out_png
