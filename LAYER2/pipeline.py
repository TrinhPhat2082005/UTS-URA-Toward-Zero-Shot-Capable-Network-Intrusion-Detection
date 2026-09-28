"""
Duong di mot sample cho Layer 2.

Khac voi cac thi nghiem truoc (prepare_evidence_*.py dung anh + context cho ca
lo truoc roi moi chay VLM), o day moi sample di het chuoi ngay trong luc chay:

    row -> Layer 1 proba -> SHAP vector -> RAG k-NN -> 5 anh -> VLM -> report

Loi ich: doi mau hay doi prompt khong can build lai evidence, va latency do
duoc la latency that cua tung chang.

Bon lop:
    Layer1Engine    - model + scaler + TreeExplainer
    EvidenceBuilder - RAG context + 5 file PNG
    VLMRunner       - nap Qwen2.5-VL, sinh bao cao, lay confidence tu logprob
    analyze_sample  - ghep ba cai tren cho mot dong
"""
import os
import time
import json
import warnings

import numpy as np
import pandas as pd

import config


# ==========================================================================
# Layer 1
# ==========================================================================
class Layer1Engine:
    """XGBoost tang cuong + TreeExplainer.

    Duong di phai khop CHINH XAC voi train_layer1.py, neu khong model se nhin
    thay du lieu khac luc train:

        X tho (78) -> RobustScaler -> clip(+-20) -> [78 | feature tang cuong]
            AE   : 16 latent + 1 recon_error   -> 95 chieu
            SVDD : K sphere_dist + 1 d_min     -> 78+K+1 chieu

    TreeExplainer chay tren vector day du, nen SHAP tra ve cung so chieu. Vector
    do la khoa truy hoi cho RAG: chieu augmented (recon_error, d_min) mang tin
    hieu "lech khoi benign" - dung thu phan biet mot flow bat thuong that voi
    mot flow benign bi gan nhan sai.
    """

    def __init__(self, variant=None):
        import joblib
        import shap

        variant = variant or config.VARIANT
        art = config.LAYER1_ARTIFACTS[variant]

        self.variant = variant
        self.model = joblib.load(art["model"])
        self.scaler = joblib.load(art["scaler"])
        self.threshold = config.layer1_threshold(variant)

        with open(art["features"], encoding="utf-8") as f:
            self.feature_names = json.load(f)
        self.n_features = int(getattr(self.model, "n_features_in_",
                                      len(self.feature_names)))
        self.n_base = int(self.scaler.n_features_in_)

        self._load_encoder(art["encoder"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.explainer = shap.TreeExplainer(self.model)

        print(f"Layer 1 ready: {variant} | {self.n_base} base + "
              f"{self.n_features - self.n_base} augmented = {self.n_features} "
              f"feature | threshold={self.threshold:.4f}")

    # ------------------------------------------------------------------
    def _load_encoder(self, path):
        """Nap AE hoac SVDD tuy variant."""
        import sys
        # LAYER2/models/ da co san ban copy cua autoencoder.py va svdd.py nen
        # thu muc nay tu chay duoc (Kaggle chi can upload LAYER2). Van giu
        # duong dan zeroshot_benchmark lam fallback cho ban local day du.
        for p in (str(config.LAYER2_DIR),
                  os.path.join(config.PROJECT_ROOT, "zeroshot_benchmark",
                               "04_training_v2")):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

        self.ae = self.svdd = None

        if self.variant.startswith("ae"):
            import torch
            from models.autoencoder import BenignAutoencoder
            n_aug = self.n_features - self.n_base          # 16 latent + 1 error
            self.ae = BenignAutoencoder(input_dim=self.n_base,
                                        bottleneck_dim=n_aug - 1)
            self.ae.load_state_dict(torch.load(path, map_location="cpu",
                                               weights_only=True))
            self.ae.eval()
            self.torch = torch
        else:
            from models.svdd import MixtureSVDD
            K = self.n_features - self.n_base - 1          # K sphere + 1 d_min
            self.svdd = MixtureSVDD(K=K, input_dim=self.n_base, device="cpu")
            self.svdd.load(path)

    def base_feature_columns(self, df):
        """78 cot feature goc, bo metadata, giu nguyen thu tu trong CSV."""
        cols = [c for c in df.columns if c not in config.META_COLS]
        if len(cols) != self.n_base:
            raise ValueError(
                f"So feature goc khong khop: CSV co {len(cols)}, scaler can "
                f"{self.n_base}. Kiem tra META_COLS trong config.py."
            )
        return cols

    # Giu ten cu de code goi truoc do khong hong
    feature_columns = base_feature_columns

    def preprocess(self, X):
        """X tho -> scale -> clip. Khop train_layer1.py."""
        X = np.nan_to_num(np.asarray(X, dtype="float64"),
                          nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(self.scaler.transform(X), -config.CLIP_VAL, config.CLIP_VAL)

    def augment(self, Xs, batch=8192):
        """Xs da scale -> ma tran feature tang cuong."""
        if self.ae is not None:
            outs = []
            with self.torch.no_grad():
                for i in range(0, len(Xs), batch):
                    xb = self.torch.tensor(Xs[i:i + batch], dtype=self.torch.float32)
                    xh, z = self.ae(xb)
                    err = ((xb - xh) ** 2).mean(dim=1, keepdim=True)
                    outs.append(self.torch.cat([z, err], dim=1).numpy())
            return np.vstack(outs)

        d = self.svdd.all_sphere_distances(Xs)
        return np.hstack([d, d.min(axis=1, keepdims=True)])

    def build_input(self, X):
        """X tho -> ma tran dung cho XGBoost (da scale, clip, ghep augmented)."""
        Xs = self.preprocess(X)
        return np.hstack([Xs, self.augment(Xs)])

    def predict_proba(self, X):
        """Xac suat tan cong tu feature THO."""
        return self.model.predict_proba(self.build_input(X))[:, 1]

    def score(self, X):
        """Tra (proba_attack, shap_values) cho ma tran feature THO.

        SHAP co cung so chieu voi input cua model (95 voi AE, 78+K+1 voi SVDD).
        """
        Xf = self.build_input(X)
        proba = self.model.predict_proba(Xf)[:, 1]

        sv = self.explainer.shap_values(Xf)
        if isinstance(sv, list):          # mot so phien ban tra list theo class
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                  # (n, feat, class)
            sv = sv[:, :, 1]
        return proba, sv


# ==========================================================================
# Evidence: RAG + anh
# ==========================================================================
class EvidenceBuilder:
    """Sinh RAG context va 5 anh cho MOT flow, ngay tai thoi diem can."""

    def __init__(self, retriever=None):
        if retriever is None:
            import sys
            if config.LAYER2_DIR not in sys.path:
                sys.path.insert(0, config.LAYER2_DIR)
            from shap_knn_retriever import SHAPKNNRetriever

            retriever = SHAPKNNRetriever(
                npz_path=config.SHAP_POOL_NPZ,
                profiles_dir=config.ATTACK_PROFILES_DIR,
                threshold=config.RAG_ZERODAY_THRESHOLD,
                exclude_labels=config.RAG_EXCLUDE_LABELS,
            )
        self.retriever = retriever

    def retrieve(self, shap_vector):
        """SHAP vector -> ket qua k-NN + khoi text cho VLM."""
        t0 = time.perf_counter()
        res = self.retriever.retrieve_batch([shap_vector], top_k=config.RAG_TOP_K)[0]
        ctx = self.retriever.format_context_for_vlm(res)
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
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

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

        # --- anh 1: SHAP waterfall ---
        shap_png = os.path.join(out_dir, "shap_waterfall.png")
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
        plt.savefig(shap_png, dpi=110, bbox_inches="tight")
        plt.close("all")

        # --- anh 2-5: bieu do luu luong ---
        import flow_visualizer
        chart_paths = flow_visualizer.generate_flow_charts(row, out_dir)

        return [shap_png] + list(chart_paths), time.perf_counter() - t0


# ==========================================================================
# VLM
# ==========================================================================
class VLMRunner:
    """Qwen2.5-VL 7B.

    Confidence lay tu logprob cua token verdict, khong dung con so model tu
    khai. Ly do: do lai v6/v7/v13 thay confidence tu khai gan nhu luon = 0.9
    ke ca khi verdict sai, nen khong quet duoc nguong nao. Con so tu khai van
    duoc giu lai o truong confidence_stated de doi chieu.
    """

    def __init__(self, model_id=None, load_4bit=None):
        import torch
        from transformers import (
            Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig,
        )

        model_id = model_id or config.VLM_MODEL_ID
        load_4bit = config.VLM_LOAD_4BIT if load_4bit is None else load_4bit

        kw = {"torch_dtype": torch.float16, "device_map": "auto"}
        if load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        print(f"Dang nap {model_id} (4bit={load_4bit})...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kw)

        # Gioi han so vision token MOI ANH.
        #
        # Qwen2.5-VL chia anh thanh patch dong theo kich thuoc that: anh cang
        # lon cang nhieu token, va attention tang bac hai theo so token. Voi 5
        # anh o 150-200 dpi (SHAP 1350x900, chart 1200x800) thi mot lan allocate
        # doi toi ~6.9 GiB -> OOM tren GPU 16GB.
        #
        # max_pixels tinh theo don vi 28*28 (kich thuoc 1 patch).
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=config.VLM_MIN_PIXELS,
            max_pixels=config.VLM_MAX_PIXELS,
        )
        self.model.eval()
        self.torch = torch
        print("VLM san sang.")

    def generate(self, system_prompt, user_prompt, image_paths):
        """Tra (raw_text, confidence_logprob, latency_s)."""
        from qwen_vl_utils import process_vision_info

        content = [{"type": "text", "text": user_prompt}]
        for p in image_paths:
            content.append({"type": "image", "image": p})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        t0 = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=config.VLM_MAX_NEW_TOKENS,
                do_sample=False,                       # greedy
                output_scores=config.VLM_RETURN_LOGPROBS,
                return_dict_in_generate=True,
            )
        latency = time.perf_counter() - t0

        seq = out.sequences[0][inputs.input_ids.shape[1]:]
        raw = self.processor.decode(seq, skip_special_tokens=True)

        conf = None
        if config.VLM_RETURN_LOGPROBS and getattr(out, "scores", None):
            conf = self._class_logprob(seq, out.scores)

        # Giai phong ngay: chay 140 mau lien tiep thi bo nho scores/inputs tich
        # luy se day GPU den OOM o vai chuc mau dau.
        del inputs, out
        self.torch.cuda.empty_cache()

        return raw, conf, latency

    def _class_logprob(self, seq, scores):
        """Xac suat model gan cho token DAU TIEN cua ten lop du doan.

        Tim cum "Predicted class:" trong chuoi token, roi lay softmax prob cua
        token ngay sau dau ':' - do la luc model thuc su chon lop nao. Day la do
        chac chan noi tai, doc lap voi con so Confidence no tu khai.

        Luu y khi bao cao: do_sample=False (greedy) nen token duoc chon luon la
        argmax, gia tri se nam trong [1/V, 1.0] va thuc te thuong > 0.3.
        """
        try:
            toks = self.processor.tokenizer.convert_ids_to_tokens(seq.tolist())
            low = [t.lower().lstrip("Ġ▁_ ") for t in toks]

            idx = None
            for i, t in enumerate(low):
                if t.startswith("predicted"):
                    # bo qua " class" va ":" de toi token ten lop
                    for j in range(i + 1, min(i + 6, len(low))):
                        if ":" in toks[j]:
                            idx = j + 1
                            break
                    break
            if idx is None or idx >= len(scores):
                return None
            probs = self.torch.softmax(scores[idx][0].float(), dim=-1)
            return float(probs[seq[idx]].item())
        except Exception:
            return None


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
        rec["rag_top_sim"] = float(rag["matches"][0]["confidence"])
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
        rec["confidence_logprob"] = conf_lp
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


def append_jsonl(rec, path=None):
    """Ghi tung ban ghi ngay - mat ket noi giua chung khong mat du lieu da chay."""
    path = path or config.RESULTS_JSONL
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        # Thu muc ket qua read-only (vd LAYER2 nam o /kaggle/input): chuyen sang
        # /kaggle/working neu co, khong thi dung thu muc tam.
        import tempfile
        alt = "/kaggle/working" if os.path.isdir("/kaggle/working") else tempfile.gettempdir()
        path = os.path.join(alt, os.path.basename(path))
        print(f"  !! khong ghi duoc vao thu muc goc -> {path}")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
