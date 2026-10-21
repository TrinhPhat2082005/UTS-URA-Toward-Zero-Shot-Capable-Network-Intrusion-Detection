"""Sinh setup1_vlm.ipynb (chay VLM cho Setup 1 tren Kaggle).

CHAY:  python LAYER2/experiments/rq4_vlm/make_setup1_notebook.py
"""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "id": f"c{len(cells):02d}", "metadata": {},
                  "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "id": f"c{len(cells):02d}", "metadata": {},
                  "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md("""
# Setup 1 - VLM chon trong ba ung vien cua retriever (Protocol B)

Moi alert o day thuoc mot trong 14 loai da biet, deu co trong tap tham chieu R, va
Hit@3 cua retriever la 100%: nhan dung LUON nam trong ba ung vien. Cau hoi duy nhat
la VLM co chon dung hon chinh ung vien gan nhat khong - moc so sanh la do chinh xac
top-1 cua retriever, ghi san trong bundle_meta.json.

UNKNOWN khong nam trong danh sach lua chon (`allow_unknown=False`). Dua no vao se tao
ra mot dap an khong bao gio dung, va con so thu duoc se do thiet ke chu khong do mo hinh.

**Kaggle:** GPU T4, Internet bat. Add-ons > Secrets: `GITHUB_TOKEN` (Attached).
Dataset: bundle cua mot variant (`setup1_bundle/<variant>/`).
""")

md("## 0. Moi truong")

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
    OUT = "/kaggle/working/setup1_out"
else:                                   # chay thu tren may local
    LAYER2 = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    VARIANT_LOCAL = os.environ.get("SETUP1_VARIANT", "ae_xgboost")
    BUNDLE = os.path.join(LAYER2, "output", "rq4_vlm", "setup1_bundle", VARIANT_LOCAL)
    OUT = os.path.join(LAYER2, "output", "rq4_vlm", f"setup1_results_{VARIANT_LOCAL}")
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers>=4.49", "accelerate", "bitsandbytes", "qwen-vl-utils",
                    "shap", "xgboost", "joblib"], check=True)
    print("xong - Run > Restart & Clear Cell Outputs, roi chay lai tu cell 1")
''')

md("## 1. Doc bundle")

code('''
import hashlib, time
import pandas as pd
import config, pipeline, prompts
import setup1_score

manifest = [json.loads(l) for l in open(os.path.join(BUNDLE, "manifest.jsonl"), encoding="utf-8") if l.strip()]
bundle_meta = json.load(open(os.path.join(BUNDLE, "bundle_meta.json"), encoding="utf-8"))
PROFILES = {os.path.basename(p)[:-5]: json.load(open(p, encoding="utf-8"))
            for p in glob.glob(os.path.join(LAYER2, "knowledge_base", "attack_profiles", "*.json"))}
VARIANT = bundle_meta["variant"]
print(json.dumps(bundle_meta, indent=1))
print(f"{len(manifest)} alert | moc can vuot: retriever top-1 = {bundle_meta['retriever_top1_accuracy']:.1%}")
''')

md("## 2. Nap VLM\n\n`MOCK_VLM = True` chi de chay thu pipeline luu/cham diem khi khong co GPU.")

code('''
MOCK_VLM = not ON_KAGGLE

class MockVLM:
    """Tra bao cao gia, luon chon ung vien dau. Chi de kiem tra code."""
    model_id = "mock"
    def generate(self, system_prompt, user_prompt, image_paths):
        opts = [l.split("Option: ")[1].split(" (")[0] for l in user_prompt.splitlines()
                if l.startswith("#### Option: ")]
        raw = ("## 1. EXECUTIVE SUMMARY\\nmock\\n## 2. ATTACK ASSESSMENT\\n"
               f"Selected option: {opts[0]}\\nTactic: Impact\\nSeverity: High\\n"
               "Top indicators: Dst Port, Flow Duration, Fwd Pkt Len Max\\nEndpoints: not available\\n"
               "## 3. FORENSIC REASONING\\n- x\\n## 4. MITRE MAPPING\\n"
               "Tactic: Impact\\nTechnique: T1499.002\\n## 5. MITIGATIONS\\n- x\\n## 6. ANALYST NOTES\\n- x")
        return raw, None, 0.0

vlm = MockVLM() if MOCK_VLM else pipeline.VLMRunner()
''')

md("""## 3. Mot alert -> mot bao cao

Doc ky prompt va output truoc khi chay ca lo. Prompt phai liet ke dung ba ung vien,
khong co UNKNOWN, va phai co dong "Attribution usually strongest for this type" lay tu
observed_features cua dung variant dang chay.""")

code('''
TEMPLATE_HASH = hashlib.sha256((prompts.SYSTEM_PROMPT + prompts._CANDIDATE_INSTRUCTIONS_NO_UNKNOWN
                                + prompts._CANDIDATE_SCHEMA_NO_UNKNOWN).encode()).hexdigest()[:12]

SHUFFLE_SEED = 42   # hoan vi thu tu in cac lua chon; xem _option_order trong prompts.py

def run_one(r):
    user_prompt, options = prompts.build_candidate_prompt(
        r["candidates"], r["flagged"], PROFILES, r["novelty_score"], r["tau_zd"],
        variant=VARIANT, allow_unknown=False, shuffle_seed=SHUFFLE_SEED)
    imgs = [os.path.join(BUNDLE, p) for p in r["images"]]
    rec = dict(r)
    rec.update(variant=VARIANT, options=options, option_order=options,
               position_of_nearest=options.index(r["retriever_top1"]) + 1,
               shuffle_seed=SHUFFLE_SEED,
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

rec = run_one(manifest[0])
print("nhan that   :", rec["true_label"], "| top-1 retriever:", rec["retriever_top1"])
print("lua chon    :", rec["options"])
print("VLM chon    :", rec.get("selected"), "| raw:", rec.get("selected_raw"))
print("tactic      :", rec.get("tactic"), "| parse_error:", rec.get("parse_error"))
print("latency     :", round(rec.get("t_vlm_s") or 0, 1), "s")
assert "UNKNOWN" not in rec["options"], "UNKNOWN khong duoc nam trong lua chon cua Setup 1"
assert "Attribution usually strongest" not in rec["user_prompt"], \\
    "observed_features van con trong prompt - phai dat include_observed=False"
print("=" * 80); print(rec["user_prompt"]); print("=" * 80)
print(rec.get("raw_output") or rec.get("traceback"))
''')

md("""## 4. Chay ca lo

Moi alert ghi ngay mot dong vao `results.jsonl`. Dut giua chung thi chay lai cell nay:
alert da co se bo qua.""")

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
    ok = "ok" if rec.get("selected") == r["true_label"] else "  "
    print(f"[{i:3d}/{len(todo)}] {r['true_label'][:22]:<22} {ok} "
          f"-> {str(rec.get('selected') or rec.get('run_error'))[:24]:<24} "
          f"{(rec.get('t_vlm_s') or 0):5.1f}s  ({(time.time()-t0)/60:.1f} phut)")
print(f"xong -> {RESULTS}")
''')

md("## 5. Cham diem")

code('''
summary = setup1_score.score(RESULTS, BUNDLE, os.path.join(OUT, "scored"))
''')

code('''
import importlib.metadata as im
def v(p):
    try:
        return im.version(p)
    except Exception:
        return None
env = {"libraries": {p: v(p) for p in ("torch", "transformers", "bitsandbytes", "accelerate")},
       "vlm": {"model": getattr(vlm, "model_id", None), "load_4bit": config.VLM_LOAD_4BIT,
               "max_new_tokens": config.VLM_MAX_NEW_TOKENS, "temperature": config.VLM_TEMPERATURE},
       "bundle_meta": bundle_meta, "template_hash": TEMPLATE_HASH,
       "allow_unknown": False, "include_observed": False,
       "context_hash": hashlib.sha256(rec["user_prompt"].encode()).hexdigest()[:12]}
json.dump(env, open(os.path.join(OUT, "run_meta.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(env, indent=1))
''')

md("""## 6. Tai ket qua ve

Kaggle: Output > `setup1_out/` (results.jsonl, scored/, run_meta.json).
Luu vao `LAYER2/output/rq4_vlm/setup1_results_<variant>/` o may local.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup1_vlm.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"-> {out} ({len(cells)} cell)")
