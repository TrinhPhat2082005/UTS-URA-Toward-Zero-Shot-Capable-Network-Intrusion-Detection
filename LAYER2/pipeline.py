"""
Duong di mot alert qua Layer 2, va noi gom cac thanh phan cho notebook / demo live.

    row -> Layer 1 (p, SHAP)        layer1/engine.py      Layer1Engine
        -> retriever (top-3, co)    retrieval/            SHAPKNNRetriever, novelty
        -> context + 5 anh          evidence/build.py     EvidenceBuilder      } chang A (CPU)
        -> VLM -> bao cao           vlm/runner.py         VLMRunner            } chang B (GPU)
                                    prompts.py            prompt + parser

analyze_sample chay hai chang lien nhau cho MOT dong (demo live, notebook run_layer2); Setup 1/2
ghi chang A thanh bundle (evidence/bundle.py) roi chay chang B tren Kaggle. Cac ten Layer1Engine,
EvidenceBuilder, VLMRunner, append_jsonl van goi duoc qua pipeline.* nhu truoc.
"""
import os
import time

import numpy as np
import pandas as pd

import config
from evidence.build import EvidenceBuilder  # noqa: F401  (pipeline.EvidenceBuilder)
from layer1.engine import Layer1Engine      # noqa: F401  (pipeline.Layer1Engine)
from utils.io import append_jsonl           # noqa: F401  (pipeline.append_jsonl)
from vlm.runner import VLMRunner            # noqa: F401  (pipeline.VLMRunner)


# ==========================================================================
# Mot sample tu dau den cuoi
# ==========================================================================
def analyze_sample(row, l1, ev, vlm, feature_names, prompt_variant="neutral",
                   out_dir=None, keep_images=False, meta=None):
    """Chay het chuoi cho MOT dong va tra ve ban ghi ket qua.

    Args:
        row: pandas Series, mot dong val_binary (co ca cot metadata).
        l1, ev, vlm: cac engine da khoi tao.
        feature_names: 78 ten cot feature.
        prompt_variant: khoa trong prompts.PROMPT_VARIANTS.
        out_dir: noi luu 5 anh; mac dinh la thu muc tam theo sample.
        keep_images: False thi xoa anh sau khi VLM doc xong.
        meta: dict ghi kem (row_idx, set_type, label...).

    Returns:
        dict - mot ban ghi JSONL.
    """
    import shutil
    import prompts

    meta = dict(meta or {})
    if out_dir is None:
        base = config.EVIDENCE_DIR
        # Tren Kaggle, LAYER2 co the nam o /kaggle/input (read-only). Neu khong
        # ghi duoc thi rot ve thu muc tam cua he thong thay vi hong ca sample.
        try:
            os.makedirs(base, exist_ok=True)
        except OSError:
            import tempfile
            base = os.path.join(tempfile.gettempdir(), "layer2_evidence")
            os.makedirs(base, exist_ok=True)
        out_dir = os.path.join(base, f"s{meta.get('row_idx', 'x')}")

    rec = dict(meta)
    rec["prompt_variant"] = prompt_variant
    rec["run_error"] = None

    try:
        X = row[feature_names].to_numpy(dtype="float64").reshape(1, -1)

        # --- Layer 1 ---
        t0 = time.perf_counter()
        Xf = l1.build_input(X)                 # scale + clip + augmented
        proba = l1.model.predict_proba(Xf)[:, 1]
        sv = l1.explainer.shap_values(Xf)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        rec["xgb_proba"] = float(proba[0])
        rec["layer1_alert"] = bool(proba[0] >= l1.threshold)
        rec["t_layer1_s"] = time.perf_counter() - t0

        shap_vec = sv[0]

        # Top-5 dong gop: M8 (evidence grounding) va M9 (hallucinated evidence)
        # doi chieu ten feature VLM neu voi danh sach nay, khong phai chay lai SHAP.
        top = np.argsort(-np.abs(shap_vec))[:5]
        rec["shap_top_features"] = [
            {"name": l1.feature_names[i], "value": float(shap_vec[i])} for i in top
        ]

        # --- RAG ---
        rag, ctx, t_rag = ev.retrieve(shap_vec)
        rec["rag_predicted_label"] = rag["predicted_label"]
        rec["rag_is_potential_zero_day"] = bool(rag["is_potential_zero_day"])
        rec["rag_candidates"] = [m["matched_label"] for m in rag["matches"]]
        rec["rag_top_distance"] = float(rag["matches"][0]["distance"])
        rec["rag_novelty_score"] = float(rag["novelty_score"])
        rec["rag_context"] = ctx
        rec["t_rag_s"] = t_rag

        # --- anh ---
        imgs, t_img = ev.make_images(row, shap_vec, l1.feature_names, out_dir,
                                     proba[0], model_input=Xf[0])
        rec["n_images"] = len(imgs)
        rec["t_images_s"] = t_img

        # --- VLM ---
        user_prompt = prompts.build_user_prompt(ctx, variant=prompt_variant)
        raw, conf_lp, t_vlm = vlm.generate(prompts.SYSTEM_PROMPT, user_prompt, imgs)
        rec["t_vlm_s"] = t_vlm
        rec["raw_output"] = raw

        parsed = prompts.parse_report(raw)
        rec["confidence_stated"] = parsed.pop("confidence")

        # conf_lp la {"p", "token"} hoac None. Ghi ca token da do de sau nay
        # doc lai duoc no da bat dung cho hay chua.
        rec["confidence_logprob"] = conf_lp["p"] if conf_lp else None
        rec["conf_token"] = conf_lp["token"] if conf_lp else None
        rec.update(parsed)

        rec["latency_s"] = (rec["t_layer1_s"] + t_rag + t_img + t_vlm)

        if not keep_images:
            shutil.rmtree(out_dir, ignore_errors=True)

    except Exception as e:
        import traceback
        rec["run_error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-1500:]

    return rec


# ==========================================================================
# Lay mau
# ==========================================================================
def load_alert_pool(l1, csv_path=None, max_rows=None):
    """Chay Layer 1 tren val_binary va tra cac dong NO DA BAO DONG.

    Alert duoc tach thanh hai nhom theo ground truth:
        tp - Label = 1 (bao dong dung)
        fp - Label = 0 (bao dong nham)
    Day la buoc Paper 4.5 mo ta.
    """
    csv_path = csv_path or config.VAL_BINARY_CSV
    df = pd.read_csv(csv_path, nrows=max_rows)
    feats = l1.base_feature_columns(df)

    proba = l1.predict_proba(df[feats].to_numpy(dtype="float64"))

    df = df.copy()
    df["_proba"] = proba
    alerts = df[df["_proba"] >= l1.threshold].copy()
    alerts["_set_type"] = np.where(alerts["Label"] == 1, "tp", "fp")

    n_tp = int((alerts["_set_type"] == "tp").sum())
    n_fp = int((alerts["_set_type"] == "fp").sum())
    print(f"{os.path.basename(csv_path)}: {len(df):,} dong -> {len(alerts):,} alert "
          f"(tp={n_tp:,}, fp={n_fp:,}) o nguong {l1.threshold:.4f}")

    # Phan bo nhan goc cua nhom FP - cho biet loai nao gay bao dong nham
    if "Label_Text" in alerts.columns and n_fp:
        fp_lab = alerts.loc[alerts["_set_type"] == "fp", "Label_Text"].value_counts()
        print(f"  FP theo nhan goc: {dict(fp_lab.head(5))}")
    return alerts, feats


def sample_stratified_tp(alerts, per_class=10, seed=None):
    """Lay <per_class> alert DUNG (tp) cho MOI loai tan cong.

    Day la thiet ke chot trong METRIC.md: nhan nho nhat (FTP_BruteForce) chi co
    10 alert, nen 10/nhan la muc cao nhat ma moi nhan deu dat -> du 14/14 nhan.
    Nhan nao it hon per_class thi lay het.

    Chi lay nhom tp: nhanh danh gia false positive da bi bo (RAG truy hoi bang
    chinh vector SHAP cua Layer 1 nen khong dung de phu dinh Layer 1 duoc).
    """
    seed = config.SAMPLE_SEED if seed is None else seed
    tp = alerts[alerts["_set_type"] == "tp"]

    parts = []
    for lab, g in tp.groupby("Label_Text"):
        k = min(per_class, len(g))
        parts.append(g.sample(k, random_state=seed))
        if k < per_class:
            print(f"  {lab:<26} chi co {len(g)} alert (can {per_class})")

    out = pd.concat(parts)
    print(f"Mau danh gia: {len(out)} alert tu {out['Label_Text'].nunique()} nhan "
          f"({per_class}/nhan, seed={seed})")
    return out.reset_index().rename(columns={"index": "_row_idx"})


def sample_balanced(alerts, n_per_group=None, seed=None):
    """Lay bang nhau moi nhom. Cap lai neu mot nhom khong du."""
    n = n_per_group or config.N_PER_GROUP
    seed = config.SAMPLE_SEED if seed is None else seed

    tp = alerts[alerts["_set_type"] == "tp"]
    fp = alerts[alerts["_set_type"] == "fp"]
    k = min(n, len(tp), len(fp))
    if k < n:
        print(f"  !! chi lay duoc {k}/nhom (tp={len(tp)}, fp={len(fp)})")

    out = pd.concat([
        tp.sample(k, random_state=seed),
        fp.sample(k, random_state=seed),
    ])
    print(f"Mau danh gia: {k} tp + {k} fp = {len(out)} (seed={seed})")
    return out.reset_index().rename(columns={"index": "_row_idx"})


# ==========================================================================
# Tai tao anh tu row_idx
# ==========================================================================
def rebuild_images(row_idx, l1, ev, feature_names, out_dir=None, val_df=None):
    """Ve lai 5 anh evidence cho MOT dong, chi can row_idx.

    5 anh khong duoc luu trong results_*.jsonl (analyze_sample xoa ngay sau
    khi VLM doc xong, keep_images=False mac dinh o cell 9) - JSON chi con lai
    2 con so n_images/t_images_s, khong mang du lieu de dung lai anh. Doc
    row_idx tu ban ghi JSON roi goi ham nay la ra dung lai duoc ca 5 anh, vi
    Layer 1 (cung checkpoint, cung threshold) tren dung dong val_full.csv do
    luon cho dung mot SHAP vector - khong co gi ngau nhien trong buoc nay.

    Args:
        row_idx: so nguyen - lay tu rec["row_idx"] cua mot dong trong
                 results_*.jsonl.
        l1, ev: Layer1Engine va EvidenceBuilder da khoi tao (nhu o cell 3-4).
        feature_names: 78 ten cot feature goc (tra ve tu load_alert_pool()).
        out_dir: noi luu anh; mac dinh EVIDENCE_DIR/rebuilt_s<row_idx>.
        val_df: DataFrame val_full.csv da doc san, de tranh doc lai file moi
                lan goi ham nay cho nhieu row_idx lien tiep.

    Returns:
        (image_paths, row) - duong dan 5 anh va dong flow dung de ve.
    """
    if val_df is None:
        val_df = pd.read_csv(config.VAL_BINARY_CSV)
    row = val_df.iloc[int(row_idx)]

    if out_dir is None:
        out_dir = os.path.join(config.EVIDENCE_DIR, f"rebuilt_s{row_idx}")

    X = row[feature_names].to_numpy(dtype="float64").reshape(1, -1)
    Xf = l1.build_input(X)
    proba, sv = l1.score(X)

    imgs, _ = ev.make_images(row, sv[0], l1.feature_names, out_dir,
                             float(proba[0]), model_input=Xf[0])
    return imgs, row
