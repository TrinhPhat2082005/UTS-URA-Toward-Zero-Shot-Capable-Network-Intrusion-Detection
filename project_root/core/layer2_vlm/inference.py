import os
import glob
ai_env_bin = r"C:\Users\TNP\miniconda3\envs\ai_env\Library\bin"
if os.path.exists(ai_env_bin):
    os.add_dll_directory(ai_env_bin)
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHAP_DIR  = os.path.join(PROJECT_ROOT, "data", "vlm_dataset", "shap_img")
CHART_DIR = os.path.join(PROJECT_ROOT, "data", "vlm_dataset", "chart")
CHART_FILES = [
    "chart_traffic_volume.png",
    "chart_protocol_distribution.png",
    "chart_src_dst_flow.png",
    "chart_port_activity.png",
]

SYSTEM_PROMPT = """\
You are an expert Tier-3 SOC (Security Operations Center) Analyst and Network Forensics \
Specialist embedded in an Agentic Intrusion Detection System (IDS). Your mission is to \
analyse multimodal evidence — machine-learning signals, network traffic visualisations, \
and retrieved threat intelligence — and produce a single, authoritative forensic report \
about a detected network security event.

You will receive the following inputs in this order:
  1. This system prompt (context and instructions).
  2. A RAG CONTEXT block — semantically retrieved profiles from a MITRE ATT&CK knowledge \
base (injected by the pipeline after this prompt).
  3. Up to 5 IMAGES — visual evidence attached to this message.

SECTION A — VISUAL INPUTS (SILENT ANALYSIS — DO NOT PRINT)
All images are INTERNAL EVIDENCE ONLY.
You MUST NOT output raw chart values, axis tick labels, legend text, pixel-level \
descriptions, or any verbatim data extraction from the images.
Instead, internalise the semantic meaning of each image as described below, \
then use that understanding silently when reasoning in your final report.

  IMAGE 1 — SHAP Waterfall Chart:
    Internally read: the predicted attack class (chart title), the peer/attacker IP \
address (chart subtitle or annotation), the top contributing features (positive SHAP = \
pushes toward attack class; negative SHAP = pushes toward benign), and the overall \
model prediction score. Identify which traffic features drove the alert.

  IMAGE 2 — Traffic Volume Chart:
    Internally read: volumetric trends over time — sudden burst spikes, sustained \
high-rate floods, or periodic pulses. Use this to assess attack intensity and timing.

  IMAGE 3 — Protocol Distribution Chart:
    Internally read: dominant protocol(s) — TCP, UDP, ICMP, HTTP, etc. Unusual \
single-protocol dominance (e.g., pure UDP or raw SYN) is a strong attack indicator.

  IMAGE 4 — Source-to-Destination Flow Chart:
    Internally read: whether traffic concentrates on one victim (DoS/DDoS) or spreads \
across many targets (scanning/brute-force). Extract the most active attacker→victim \
IP pairs.

  IMAGE 5 — Port Activity Chart:
    Internally read: top targeted destination ports and their associated services. \
Map port numbers to known attack vectors:
      port 22   → SSH brute-force (T1110.001)
      port 80/443 → HTTP/HTTPS flood (T1498.002)
      port 21   → FTP brute-force (T1110.001)
      high ephemeral → reflection/amplification (T1498.001)

After silently reading all images, synthesise the visual evidence with the RAG context \
to produce only the structured report defined in Section C below.

SECTION B — RAG CONTEXT (ATTACK PROFILE KNOWLEDGE BASE)
The following block is retrieved from a FAISS-indexed MITRE ATT&CK knowledge base. \
It contains attack pattern signatures, typical ML feature indicators, tactic chains, \
and recommended mitigations. Use this as your PRIMARY reference for:
  - Identifying which known attack pattern best matches the observed evidence.
  - Selecting precise MITRE ATT&CK Technique / Sub-technique IDs.
  - Deriving specific, evidence-based mitigation recommendations.
  - Cross-validating ML model output against historical attack profiles.

[RAG CONTEXT IS INJECTED BELOW THIS LINE BY THE PIPELINE]

SECTION C — OUTPUT: PROFESSIONAL FORENSIC REPORT
Produce ONLY the following markdown sections, in this exact order.
Do not add any free-form text, preamble, or explanation outside these sections.
Do not repeat or paraphrase image data — derive insights only.

## EXECUTIVE SUMMARY

Write 3–4 concise sentences covering:
(1) What the ML model detected and its confidence level.
(2) Whether the real-traffic visual evidence corroborates or contradicts the ML verdict.
(3) Which MITRE ATT&CK attack profile from the RAG knowledge base best matches the event.
(4) The overall analyst conclusion (confirmed threat or false positive).

##ATTACK ASSESSMENT

| Field                    | Assessment                                                       |
|--------------------------|------------------------------------------------------------------|
| **Verdict**              | CONFIRMED ATTACK  /  FALSE POSITIVE                              |
| **Confidence**           | XX%                                                              |
| **Predicted Class**      | (attack type label from ML model)                                |
| **Attacker IP**          | (source IP derived from visual & RAG evidence)                   |
| **Victim IP / Port**     | (destination IP and port from evidence)                          |
| **Attack Vector**        | (transport protocol and method, e.g., TCP SYN flood)             |
| **Traffic Signature**    | (key behavioural pattern: burst, sustained flood, scatter, etc.) |
| **Top ML Indicators**    | (top 3 SHAP features driving the prediction)                     |
| **RAG Pattern Match**    | (best-matching profile name from RAG context)                    |
| **Severity**             | Critical  /  High  /  Medium  /  Low                             |

##FORENSIC REASONING

Use structured bullet points. Cross-validate ALL evidence sources:

- **ML Signal:** What the SHAP chart reveals about which features drove the alert and why.
- **Volumetric Pattern:** Traffic load characteristics — burst, sustained, or periodic.
- **Protocol Anomaly:** Whether the protocol mix deviates from normal baseline expectations.
- **Flow Concentration:** Whether the attack targets a single victim (DoS/DDoS) or is distributed (scanning/brute-force).
- **Service Targeting:** Which service or application is being attacked based on port activity.
- **RAG Corroboration:** How the retrieved MITRE profile supports or challenges the observed evidence.
- **Convergence / Contradiction:** State whether all evidence lines converge on a single conclusion, or identify contradictions that reduce confidence.

##MITRE ATT&CK MAPPING

*(Complete this section ONLY if Verdict = CONFIRMED ATTACK. Otherwise write: "N/A — False Positive.")*

| Field              | Details                                                          |
|--------------------|------------------------------------------------------------------|
| **Tactic**         | (e.g., Impact / Initial Access / Credential Access / Discovery)  |
| **Technique ID**   | (e.g., T1498)                                                    |
| **Technique Name** | (full technique name)                                            |
| **Sub-technique**  | (e.g., T1498.001 — Direct Network Flood, or N/A)                 |
| **Data Sources**   | (telemetry or logs required to detect this technique)            |
| **Detection Note** | (brief note on how this technique is detected in practice)       |

##MITIGATION RECOMMENDATIONS

*(Complete this section ONLY if Verdict = CONFIRMED ATTACK. Otherwise write: "N/A — False Positive.")*

Provide a prioritised numbered list of concrete mitigations. Each item MUST:
  - Reference a MITRE ATT&CK Mitigation ID (e.g., M1037) where available.
  - Be specific to the indicators observed — not generic advice.
  - Indicate the target system, service, or responsible team.

Format:
  1. **[MXXXX] Mitigation Name** — Specific action targeting the observed indicator, e.g., block source IP range X.X.X.X/YY at the perimeter firewall.
  2. **[MXXXX] Mitigation Name** — …
  3. (add as many as needed, in priority order)

##ANALYST NOTES

Flag any of the following that apply (omit sub-items that do not apply):
  - **Data Gaps:** Missing images or telemetry that limit analysis confidence.
  - **Further Investigation:** Recommended next steps (e.g., full pcap capture, threat hunting, endpoint correlation).
  - **Caveats:** Any ambiguity in the evidence that downstream analysts should be aware of.

*Report generated by Agentic VLM IDS — Layer 2 Forensic Analyst | Qwen2.5-VL + MITRE ATT&CK RAG*
"""


def get_shap_paths(shap_dir: str = SHAP_DIR) -> list[str]:
    pattern = os.path.join(shap_dir, "SHAP_img_*.png")
    paths = sorted(
        glob.glob(pattern),
        key=lambda p: int(
            os.path.splitext(os.path.basename(p))[0].replace("SHAP_img_", "")
        )
    )
    if paths:
        print(f"[SHAP] Found {len(paths)} images in: {shap_dir}")
    else:
        print(f"[SHAP] No images found in: {shap_dir}")
    return paths


def get_chart_paths(chart_dir: str = CHART_DIR) -> list[str]:
    paths = []
    for fname in CHART_FILES:
        full = os.path.join(chart_dir, fname)
        if os.path.exists(full):
            paths.append(full)
            print(f"[Chart] Loaded: {fname}")
        else:
            print(f"[Chart] Missing file: {full}")
    return paths


def _embed_images(content: list, img_paths: list[str], group_label: str) -> int:
    valid = 0
    for img_path in img_paths:
        abs_path = os.path.abspath(img_path)
        if os.path.exists(abs_path):
            content.append({"type": "image", "image": f"file://{abs_path}"})
            print(f"-> [{group_label}] {os.path.basename(abs_path)}")
            valid += 1
        else:
            print(f"[Warning] Image not found: {abs_path}")
    return valid


class VLMAnalyst:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not detected. Please verify driver and PyTorch installation.")
        self.device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        # Set memory allocation configuration to avoid fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        print(f"[Layer 2] GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")
        print(f"[Layer 2] Loading VLM: {model_id} (4-bit quantized)...")

        quantation_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
            quantization_config=quantation_config,
        )
        self.model.eval()

        # Limit image resolution to save VRAM (Qwen2-VL specific)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=256*28*28,
            max_pixels=512*28*28
        )
        print("[Layer 2] VLM loaded successfully.")

    def final_reasoning(
        self,
        shap_paths: list[str],
        chart_paths: list[str],
        rag_context: str = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: int = 1500,
    ) -> str:
        """
        Multimodal inference with memory optimizations.
        Combines SHAP images, traffic charts, and RAG context
        to produce a structured professional forensic report.
        """
        # Clear cache before starting
        torch.cuda.empty_cache()

        content: list = [
            {"type": "text", "text": system_prompt},
        ]

        # Inject RAG context immediately after system prompt
        if rag_context:
            content.append({
                "type": "text",
                "text": f"\n[RAG CONTEXT — RETRIEVED ATTACK PROFILES]\n{rag_context}\n[END OF RAG CONTEXT]\n"
            })

        n_shap  = _embed_images(content, shap_paths,  "SHAP")
        n_chart = _embed_images(content, chart_paths, "Chart")

        print(f"[Layer 2] Evidence loaded: {n_shap} SHAP image(s), {n_chart} chart(s).")

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        print("[Layer 2] VLM forensic reasoning in progress...")
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.1,
                    do_sample=False,  # Deterministic for reduced memory spikes
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return response[0]
        finally:
            # Clean up GPU memory after inference
            del inputs
            if 'image_inputs' in locals():
                del image_inputs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    print("[VLM] Independent test session starting...")
    shap_paths  = get_shap_paths()
    chart_paths = get_chart_paths()
    vlm = VLMAnalyst()
    result = vlm.final_reasoning(
        shap_paths=shap_paths,
        chart_paths=chart_paths,
    )
    print("\n" + "="*60)
    print("FORENSIC ANALYSIS REPORT")
    print("="*60)
    print(result)
