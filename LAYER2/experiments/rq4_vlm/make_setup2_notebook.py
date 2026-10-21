"""Sinh setup2_vlm.ipynb (chay VLM cho Setup 2 tren Kaggle).

CHAY:  python LAYER2/experiments/rq4_vlm/make_setup2_notebook.py
"""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "id": f"c{len(cells):02d}", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "id": f"c{len(cells):02d}", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md("""
# Setup 2 — VLM bao cao loai tan cong chua tung thay (Protocol A)

Moi fold leave-one-attack-out giu lai mot loai h. Layer 1, tap tham chieu R va tau_zd
deu la cua fold do; h khong co trong R va khong co trong danh sach lua chon, nen dap an
dung cua MOI alert la `UNKNOWN`.

Evidence (Layer 1, TreeSHAP, retriever, 5 anh) da dung san tren may local bang
`experiments/rq4_vlm/setup2_build_evidence.py`. Notebook nay chi chay VLM tren bundle do,
luu toan bo prompt va output, roi cham diem bang `setup2_score.py`.

Lua chon cua VLM: 3 loai gan nhat + `UNKNOWN`.
- Co co zero-day: profile unknown_anomaly va canh bao dat len dau, roi 3 ung vien.
- Khong co co: 3 ung vien, roi `UNKNOWN` o cuoi (khong profile).

**Kaggle:** GPU T4, Internet bat. Add-ons > Secrets: `GITHUB_TOKEN` (Attached).
Add Input: dataset `layer2-setup2-bundle`. Chay tu tren xuong; cell 1 tu nhan biet Kaggle.
""")

code('''
import os, sys, glob, json, subprocess

ON_KAGGLE = os.path.isdir("/kaggle/input")
if ON_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
    REPO = f"https://{TOKEN}@github.com/TrinhPhat2082005/UTS-URA-Toward-Zero-Shot-Capable-Network-Intrusion-Detection.git"
    CODE = "/kaggle/working/UTS-URA"
    if not os.path.isdir(CODE):
        subprocess.run(["git", "clone", "--depth", "1", REPO, CODE], check=True)
    else:
        subprocess.run(["git", "-C", CODE, "pull"], check=True)
    LAYER2 = os.path.join(CODE, "LAYER2")
    BUNDLE = os.path.dirname(glob.glob("/kaggle/input/**/manifest.jsonl", recursive=True)[0])
    OUT = "/kaggle/working/setup2_out"
else:                                   # chay thu tren may local
    LAYER2 = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    BUNDLE = os.path.join(LAYER2, "output", "rq4_vlm", "setup2_bundle")
    OUT = os.path.join(LAYER2, "output", "rq4_vlm", "setup2_results")
os.makedirs(OUT, exist_ok=True)
for p in (LAYER2, os.path.join(LAYER2, "experiments", "rq4_vlm")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(LAYER2)
print("code  :", LAYER2)
print("bundle:", BUNDLE)
print("out   :", OUT)
''')

md("Cai dependency (chi Kaggle). Cai xong bitsandbytes phai **restart kernel** roi chay lai tu cell 1.")

code('''
if ON_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "bitsandbytes>=0.46.1", "qwen-vl-utils"], check=True)
''')

md("## 1. Nap bundle")

code('''
import hashlib, time, platform
import numpy as np, pandas as pd
import config, prompts, pipeline
import setup2_score

config.VLM_RETURN_LOGPROBS = False        # prompt moi khong co truong "Predicted class"
PROFILES = {}
for f in glob.glob(os.path.join(config.ATTACK_PROFILES_DIR, "*.json")):
    d = json.load(open(f, encoding="utf-8")); PROFILES[d["id"]] = d

manifest = [json.loads(l) for l in open(os.path.join(BUNDLE, "manifest.jsonl"), encoding="utf-8")]
bundle_meta = json.load(open(os.path.join(BUNDLE, "bundle_meta.json"), encoding="utf-8"))
fold_stats = json.load(open(os.path.join(BUNDLE, "fold_stats.json"), encoding="utf-8"))
VARIANT = bundle_meta["variant"]
print(json.dumps(bundle_meta, indent=1))
cols = ["n_test", "n_alerts", "zs_acc", "tau_alpha", "tau_zd", "flag_rate", "n_sampled",
        "n_sampled_flagged", "excluded"]
fs = pd.DataFrame(fold_stats).T
print(fs[[c for c in cols if c in fs]].to_string())
print(f"{len(manifest)} alert, {sum(r['flagged'] for r in manifest)} co co")
''')

md("## 2. Nap VLM\n\n`MOCK_VLM = True` chi de chay thu pipeline luu/cham diem khi khong co GPU.")

code('''
MOCK_VLM = not ON_KAGGLE

class MockVLM:
    """Tra bao cao gia, chon UNKNOWN khi co co, top-1 khi khong co. Chi de kiem tra code."""
    model_id = "mock"
    def generate(self, system_prompt, user_prompt, image_paths):
        opts = [l.split("Option: ")[1].split(" (")[0] for l in user_prompt.splitlines()
                if l.startswith("#### Option: ")]
        sel = "UNKNOWN" if "NOVELTY FLAG RAISED" in user_prompt else opts[0]
        tech = "N/A" if sel == "UNKNOWN" else "T1499.002"
        raw = ("## 1. EXECUTIVE SUMMARY\\nmock\\n## 2. ATTACK ASSESSMENT\\n"
               f"Selected option: {sel}\\nTactic: Impact\\nSeverity: High\\n"
               "Top indicators: Dst Port, Flow Duration, Fwd Pkt Len Max\\nEndpoints: not available\\n"
               "## 3. FORENSIC REASONING\\n- x\\n## 4. MITRE MAPPING\\n"
               f"Tactic: Impact\\nTechnique: {tech}\\n## 5. MITIGATIONS\\n- x\\n## 6. ANALYST NOTES\\n- x")
        return raw, None, 0.0

vlm = MockVLM() if MOCK_VLM else pipeline.VLMRunner()
''')

md("## 3. Mot alert -> mot bao cao\n\nDoc ky prompt va output truoc khi chay ca lo.")

code('''
TEMPLATE_HASH = hashlib.sha256((prompts.SYSTEM_PROMPT + prompts._CANDIDATE_INSTRUCTIONS
                                + prompts._CANDIDATE_SCHEMA).encode()).hexdigest()[:12]
SHUFFLE_SEED = 42   # hoan vi thu tu in cac lua chon; xem _option_order trong prompts.py

def run_one(r):
    user_prompt, options = prompts.build_candidate_prompt(
        r["candidates"], r["flagged"], PROFILES, r["novelty_score"], r["tau_zd"],
        variant=VARIANT, shuffle_seed=SHUFFLE_SEED)
    imgs = [os.path.join(BUNDLE, p) for p in r["images"]]
    rec = dict(r)
    rec.update(options=options, option_order=options, shuffle_seed=SHUFFLE_SEED,
               position_of_nearest=options.index(r["retriever_top1"]) + 1,
               user_prompt=user_prompt, template_hash=TEMPLATE_HASH,
               vlm_model=getattr(vlm, "model_id", config.VLM_MODEL_ID), run_error=None,
               started_at=pd.Timestamp.now().isoformat(timespec="seconds"))
    try:
        raw, _, t = vlm.generate(prompts.SYSTEM_PROMPT, user_prompt, imgs)
        rec.update(raw_output=raw, t_vlm_s=t)
        rec.update(prompts.parse_candidate_report(raw, options))
    except Exception as e:
        import traceback
        rec.update(run_error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
    return rec

ex = next(r for r in manifest if r["flagged"])
rec = run_one(ex)
print("fold / co   :", rec["fold"], "/", rec["flagged"], "| top-1 retriever:", rec["retriever_top1"])
print("lua chon    :", rec["options"])
print("VLM chon    :", rec.get("selected"), "| raw:", rec.get("selected_raw"))
print("tactic      :", rec.get("tactic"), "| that:", rec["true_tactics"])
print("technique   :", rec.get("technique_ids"), "| parse_error:", rec.get("parse_error"))
print("latency     :", round(rec.get("t_vlm_s") or 0, 1), "s")
print("=" * 80); print(rec["user_prompt"]); print("=" * 80); print(rec.get("raw_output") or rec.get("traceback"))
''')

md("""## 4. Chay ca lo

Moi alert ghi ngay mot dong vao `results.jsonl` (day du: manifest, prompt, output tho,
truong da parse, thoi gian). Dut giua chung thi chay lai cell nay: alert da co se bo qua.""")

code('''
RESULTS = os.path.join(OUT, "results.jsonl")
done = set()
if os.path.exists(RESULTS):
    done = {json.loads(l)["alert_id"] for l in open(RESULTS, encoding="utf-8") if l.strip()}
    print(f"da co {len(done)} ban ghi - bo qua")

t0 = time.time()
todo = [r for r in manifest if r["alert_id"] not in done]
for i, r in enumerate(todo, 1):
    rec = run_one(r)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
    print(f"[{i:3d}/{len(todo)}] {r['fold'][:22]:<22} co={int(r['flagged'])} "
          f"-> {str(rec.get('selected') or rec.get('run_error'))[:24]:<24} "
          f"{(rec.get('t_vlm_s') or 0):5.1f}s  ({(time.time()-t0)/60:.1f} phut)")
print(f"xong -> {RESULTS}")
''')

md("## 5. Cham diem + luu ket qua")

code('''
summary, per_fold = setup2_score.score(RESULTS, BUNDLE, os.path.join(OUT, "scored"))
''')

code('''
import importlib.metadata as im
def ver(p):
    try:
        return im.version(p)
    except Exception:
        return None
gpu = None
try:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception:
    pass
commit = subprocess.run(["git", "-C", LAYER2, "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
run_meta = {
    "setup": "Setup 2", "mock_vlm": MOCK_VLM, "vlm_model": config.VLM_MODEL_ID,
    "load_4bit": config.VLM_LOAD_4BIT, "max_new_tokens": config.VLM_MAX_NEW_TOKENS,
    "min_pixels": config.VLM_MIN_PIXELS, "max_pixels": config.VLM_MAX_PIXELS, "decoding": "greedy",
    "template_hash": TEMPLATE_HASH, "code_commit": commit, "bundle_meta": bundle_meta,
    "gpu": gpu, "python": platform.python_version(),
    "versions": {p: ver(p) for p in ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "xgboost")},
    "n_results": sum(1 for _ in open(RESULTS, encoding="utf-8")),
    "saved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
}
json.dump(run_meta, open(os.path.join(OUT, "run_meta.json"), "w"), indent=1)
with open(os.path.join(OUT, "prompt_template.txt"), "w", encoding="utf-8") as f:
    f.write("=== SYSTEM ===\\n" + prompts.SYSTEM_PROMPT + "\\n\\n=== INSTRUCTIONS ===\\n"
            + prompts._CANDIDATE_INSTRUCTIONS + "\\n\\n=== SCHEMA ===\\n" + prompts._CANDIDATE_SCHEMA)

import shutil
zip_path = shutil.make_archive(OUT, "zip", OUT)
print("da luu:", sorted(os.listdir(OUT)))
print("file nen de tai ve:", zip_path)
print("Kaggle: bam 'Save Version' de giu /kaggle/working sau khi tat session.")
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup2_vlm.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"wrote {out} ({len(cells)} cells)")
