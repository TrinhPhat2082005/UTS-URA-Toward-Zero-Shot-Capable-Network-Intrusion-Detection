"""
Chang B cua Layer 2 (GPU): prompt + 5 anh -> bao cao cua Qwen2.5-VL 7B.

Chi doc evidence (pipeline live) hoac bundle (Setup 1/2), khong can Layer 1 hay tap R.
Prompt va parser nam o prompts.py.
"""
import time

import config


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
        """Tra (raw_text, conf, latency_s).

        conf = {"p": float, "token": str} hoac None neu khong xac nhan duoc
        vi tri token ten lop.
        """
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
            # Parse ten lop TRUOC roi moi do confidence: phai biet model da
            # chon lop gi thi moi xac nhan duoc token do dung la token do.
            import prompts
            cls = prompts.parse_report(raw).get("predicted_class")
            conf = self._class_logprob(seq, out.scores, cls)

        # Giai phong ngay: chay 140 mau lien tiep thi bo nho scores/inputs tich
        # luy se day GPU den OOM o vai chuc mau dau.
        del inputs, out
        self.torch.cuda.empty_cache()

        return raw, conf, latency

    # So token toi da duoc phep bo qua sau dau ':' truoc khi tu bo. Du cho cac
    # ky tu dem nhu " ", "**", "\n"; qua nguong nay thi chac chan da lech.
    _CONF_MAX_SKIP = 5

    def _class_logprob(self, seq, scores, predicted_class):
        """Xac suat model gan cho token DAU TIEN cua ten lop.

        Tra {"p": float, "token": str} hoac None.

        VI SAO CAN predicted_class
        --------------------------
        Chi dinh vi bang phep cong don do dai token la khong du: lech mot token
        thi p van la mot so hop le (xac suat cua dau ':' chang han) ma khong co
        gi bao loi - sai am tham. Nen sau khi dinh vi, doi chieu token voi ten
        lop ma parser da boc duoc tu text; hai duong nay doc lap nhau nen khop
        duoc la bang chung du manh.

        Khong yeu cau ten lop thuoc 14 nhan: neu VLM tra "DoS GoldenEye Attack"
        thi token "DoS" van khop prefix, van do duoc. Viec ten do co dung
        taxonomy hay khong la chuyen cua M1, khong phai cua ham nay.

        Luu y khi bao cao: do_sample=False (greedy) nen token duoc chon luon la
        argmax; gia tri nam trong [1/V, 1.0]. Ten lop dai bi cat nhieu token
        nen chi do duoc token dau (vd "Do" cua "DoS_attacks_Hulk").
        """
        cls = (predicted_class or "").strip().strip("*`").lower()
        if not cls:
            return None

        try:
            import re
            tok = self.processor.tokenizer
            ids = seq.tolist()
            text = tok.decode(ids, skip_special_tokens=False)

            # Muc 2 moi la cho model chon lop; chu "predicted" cung xuat hien o
            # muc 1 (Executive summary) nen phai neo theo tieu de muc 2 truoc.
            m2 = re.search(r"#{1,4}\s*2\.", text)
            start = m2.end() if m2 else 0

            m = re.compile(r"predicted\s*class\s*\**\s*:", re.I).search(text, start)
            if m is None:
                return None
            target = m.end()          # ngay sau dau ':'

            # Cong don do dai token da decode de doi vi tri KY TU sang chi so
            # TOKEN. Cach nay khong phu thuoc tokenizer cat chu the nao.
            acc, first = 0, None
            for i, t in enumerate(ids):
                acc += len(tok.decode([t], skip_special_tokens=False))
                if acc > target:
                    first = i
                    break
            if first is None:
                return None

            # Tu vi tri do, bo qua cac token dem (" ", "**", xuong dong) cho
            # den khi gap token bat dau bang CHU, roi xac nhan no la doan dau
            # cua ten lop.
            for i in range(first, min(first + self._CONF_MAX_SKIP, len(ids))):
                if i >= len(scores):
                    return None
                raw_tok = tok.decode([ids[i]], skip_special_tokens=False)
                t = raw_tok.strip().strip("*`")
                if not t or not t[0].isalpha():
                    continue                      # ky tu dem, doc tiep
                if not cls.startswith(t.lower()):
                    return None                   # chu la -> da lech, bo
                probs = self.torch.softmax(scores[i][0].float(), dim=-1)
                return {"p": float(probs[ids[i]].item()), "token": raw_tok}
            return None
        except Exception:
            return None
