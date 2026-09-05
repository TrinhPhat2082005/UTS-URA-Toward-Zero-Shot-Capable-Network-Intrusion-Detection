"""
Clean if-else reasoning workflow — runs ONLY on the remote Kaggle GPU. Uses
gate_prompt.py's CLEAN_REASONING_PROMPT: read evidence -> reason -> decide
ATTACK/BENIGN -> if ATTACK, pick a RAG candidate and write a short report.
No few-shot example (kept minimal on purpose).

Usage (on the Kaggle box):
    python3 run_vlm_remote_clean.py --evidence_dir evidence --out results_clean.jsonl
"""
import os
import sys
import json
import time
import glob
import argparse

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from gate_prompt import CLEAN_REASONING_PROMPT, parse_clean_reasoning_decision  # noqa: E402


def load_model(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not detected on this remote box.")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"[VLM] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[VLM] Loading {model_id} (4-bit quantized)...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="sdpa", quantization_config=quant_config,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
    )
    print("[VLM] Loaded.")
    return model, processor


def run_one(model, processor, shap_path, chart_paths, rag_context=None, max_new_tokens=2000):
    torch.cuda.empty_cache()
    content = [{"type": "text", "text": CLEAN_REASONING_PROMPT}]
    if rag_context:
        content.append({
            "type": "text",
            "text": f"\n[RAG CONTEXT — RETRIEVED ATTACK PROFILES]\n{rag_context}\n[END OF RAG CONTEXT]\n"
        })
    for p in [shap_path] + chart_paths:
        if p and os.path.exists(p):
            content.append({"type": "image", "image": f"file://{os.path.abspath(p)}"})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                repetition_penalty=1.1, do_sample=False,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
        response = processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
        return response[0]
    finally:
        del inputs
        torch.cuda.empty_cache()


def iter_evidence_bundles(evidence_dir, limit=None):
    n_yielded = 0
    for set_type in ["fp", "tp"]:
        set_dir = os.path.join(evidence_dir, set_type)
        if not os.path.isdir(set_dir):
            continue
        for row_dir in sorted(glob.glob(os.path.join(set_dir, "*")),
                               key=lambda p: int(os.path.basename(p))):
            if limit is not None and n_yielded >= limit:
                return
            meta_path = os.path.join(row_dir, "meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            shap_path = os.path.join(row_dir, meta["shap_image"]) if meta.get("shap_image") else None
            chart_paths = [os.path.join(row_dir, c) for c in meta.get("chart_images", [])]
            yield meta, shap_path, chart_paths, meta.get("rag_context")
            n_yielded += 1


def main():
    parser = argparse.ArgumentParser(description="Run clean if-else VLM reasoning over evidence bundles (Kaggle GPU only)")
    parser.add_argument("--evidence_dir", default="evidence")
    parser.add_argument("--out", default="results_clean.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N evidence bundles (for small test runs).")
    args = parser.parse_args()

    model, processor = load_model()

    n_done = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for meta, shap_path, chart_paths, rag_context in iter_evidence_bundles(args.evidence_dir, limit=args.limit):
            t0 = time.perf_counter()
            try:
                raw_output = run_one(model, processor, shap_path, chart_paths, rag_context=rag_context)
                pred = parse_clean_reasoning_decision(raw_output)
                error = None
            except Exception as e:
                raw_output = None
                pred = {"verdict": None, "predicted_attack_type": None,
                        "predicted_mitre_id": None, "chosen_rag_rank": None,
                        "confidence": None, "report": None, "reasoning": None,
                        "parse_error": None}
                error = f"{type(e).__name__}: {e}"
            latency_s = time.perf_counter() - t0

            row = {
                "row_idx": meta["row_idx"],
                "set_type": meta["set_type"],
                "label": meta["label"],
                "xgb_proba": meta["xgb_proba"],
                "rag_predicted_label": meta.get("rag_predicted_label"),
                "rag_is_potential_zero_day": meta.get("rag_is_potential_zero_day"),
                "rag_context": meta.get("rag_context"),
                "verdict": pred["verdict"],
                "predicted_attack_type": pred["predicted_attack_type"],
                "predicted_mitre_id": pred["predicted_mitre_id"],
                "chosen_rag_rank": pred["chosen_rag_rank"],
                "confidence": pred["confidence"],
                "report": pred["report"],
                "reasoning": pred["reasoning"],
                "parse_error": pred["parse_error"],
                "run_error": error,
                "raw_output": raw_output,
                "latency_s": latency_s,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            n_done += 1
            print(f"[Progress] {n_done} processed ({meta['set_type']}/{meta['row_idx']}, "
                  f"{latency_s:.1f}s, verdict={pred['verdict']}, parse_error={pred['parse_error']})")

    print(f"\nDone. Processed {n_done} rows. Output: {args.out}")


if __name__ == "__main__":
    main()
