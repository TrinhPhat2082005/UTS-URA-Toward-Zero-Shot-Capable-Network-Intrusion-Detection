"""Sinh vlm_run.ipynb (notebook duy nhat chay VLM). Chay lai sau moi lan sua: python LAYER2/make_vlm_notebook.py"""
import json
import os

C = []


def _lines(s):
    lines = s.strip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + lines[-1:]


def md(s):
    C.append({"cell_type": "markdown", "id": f"cell-{len(C):02d}", "metadata": {}, "source": _lines(s)})


def code(s):
    C.append({"cell_type": "code", "id": f"cell-{len(C):02d}", "execution_count": None, "metadata": {},
              "outputs": [], "source": _lines(s)})


md("""# Layer 2 — chay toan bo pipeline cho tung sample (Layer 1 → tra cuu → KB → VLM)

Moi sample la mot dong flow **tho** (samples.csv). Notebook chay lai tat ca, do latency tung chang va ghi:

```
<RUN_DIR>/<sample_id>/evidence/   5 anh + top5.json + evidence.json (tra cuu, bang C, doan KB)
<RUN_DIR>/<sample_id>/text/       system_prompt.txt, user_prompt.txt (prompt day du truoc khi gui VLM), contexts.json
<RUN_DIR>/<sample_id>/output/     report.md (nguyen van VLM), report.json (parse + trich dan)
<RUN_DIR>/<sample_id>/record.json latency tung chang
<RUN_DIR>/results.jsonl · ragas.jsonl (user_input, retrieved_contexts, response) · latency.csv · run_meta.json
```

Chang do latency (giay, `time.perf_counter`, mot flow moi lan): `layer1` (scale + AE + XGBoost) · `shap` ·
`lookup` (5-NN + novelty) · `contrast` (chi khi co bat) · `kb_select` · `images` · `prompt` · `vlm` · `parse`.

Chay tu tren xuong. Mat ket noi giua chung: chay lai cell **Chay ca lo**, sample da xong duoc bo qua.""")

md("""## 0. Setup Kaggle (bo qua khi chay local)

1. Add-ons → Secrets: secret ten `VLM_NIDS` (bien `SECRET` o cell 1) = GitHub token fine-grained, chi repo nay,
   quyen *Contents: Read-only*; tick **Attached**. Khong in token ra output.
2. Add Input: dataset tao tu `kaggle_upload/layer2-vlm-run` (samples.csv + checkpoint Layer 1 + tap R).
3. Accelerator: GPU T4 x2 (hoac P100). Chay cell cai thu vien, **restart kernel**, roi chay tu cell 1.""")

code('''# Chi chay MOT lan tren Kaggle, xong thi Restart kernel.
# Cung phien ban voi luc train / build tap R de nap duoc checkpoint va SHAP trung khop.
!pip install -q -U "bitsandbytes>=0.46.1" qwen-vl-utils "xgboost==3.2.0" "shap==0.51.0" "scikit-learn==1.8.0"''')

md("## 1. Code + du lieu")
code('''import os, sys, subprocess, glob

BRANCH = "feat/knowledge-base"
REPO = "github.com/TrinhPhat2082005/UTS-URA-Toward-Zero-Shot-Capable-Network-Intrusion-Detection.git"
SECRET = "VLM_NIDS"                  # ten Kaggle secret chua GitHub token (fine-grained, Contents: Read-only)
ON_KAGGLE = os.path.isdir("/kaggle/working")

if ON_KAGGLE:
    CODE = "/kaggle/working/UTS-URA"
    if not os.path.isdir(CODE):
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret(SECRET)
        # token o vi tri MAT KHAU (x-access-token:<token>); loi thi che token truoc khi in
        r = subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                            f"https://x-access-token:{token}@{REPO}", CODE], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("git clone loi:\\n" + (r.stderr or r.stdout).replace(token, "***"))
        # khong de token trong .git/config (/kaggle/working duoc luu cung output khi Save Version)
        subprocess.run(["git", "-C", CODE, "remote", "set-url", "origin", f"https://{REPO}"], check=True)
        del token, r
    hits = glob.glob("/kaggle/input/**/samples.csv", recursive=True)
    assert hits, "Khong thay samples.csv trong /kaggle/input: da Add Input dataset chua?"
    os.environ["UTS_DATA"] = os.path.dirname(hits[0])
    os.environ["UTS_OUT"] = "/kaggle/working"
    LAYER2 = os.path.join(CODE, "LAYER2")
else:
    # local: mo notebook tu LAYER2/; UTS_DATA tro vao thu muc co samples.csv + artifacts/
    LAYER2 = os.path.abspath(".")
    os.environ.setdefault("UTS_DATA", os.path.join(LAYER2, "..", "..", "LAYER2", "kaggle_upload", "layer2-vlm-run"))

os.environ["UTS_LAYER1_CODE"] = os.path.join(os.path.dirname(LAYER2), "LAYER1", "03_training")
os.chdir(LAYER2)
if LAYER2 not in sys.path:
    sys.path.insert(0, LAYER2)
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)''')

code('''import json, shutil, time, warnings
import pandas as pd
warnings.filterwarnings("ignore")

import config, pipeline, prompts, metrics
from evidence.build import EvidenceBuilder
assert config.describe(), "Thieu file: xem bang tren"''')

md("""## 2. Tham so lan chay

`KB_MODE = "off"` hoac bot nhom trong `KB_PARTS` de chay ablation; moi cau hinh duoc ghi vao thu muc rieng.""")
code('''KB_MODE = config.KB_MODE             # "on" | "off"
KB_PARTS = config.KB_PARTS           # ("M", "B", "H1", "H2", "H3", "G")
full = KB_MODE == "on" and set(KB_PARTS) == set(prompts.KB_ALL_PARTS)
RUN_NAME = "vlm_run" if full else f"vlm_run_{KB_MODE}_{'-'.join(KB_PARTS)}"
RUN_DIR = os.path.join(config.OUT_DIR, RUN_NAME)
N_WARMUP = 3                         # sample chay truoc (khong ghi) de nap cache truoc khi do latency
os.makedirs(RUN_DIR, exist_ok=True)
print(RUN_DIR)''')

md("## 3. Nap Layer 1, tap R, KB (CPU) va VLM (GPU)")
code('''t0 = time.perf_counter()
eb = EvidenceBuilder()
LOAD = {"evidence_builder_s": time.perf_counter() - t0}
print(f"Layer 1 + tap R ({len(eb.store):,} vector) + KB: {LOAD['evidence_builder_s']:.1f} s")''')
code('''from vlm.runner import VLMRunner
t0 = time.perf_counter()
vlm = VLMRunner()
LOAD["vlm_s"] = time.perf_counter() - t0
print(f"VLM: {LOAD['vlm_s']:.1f} s")''')

md("## 4. Sample")
code('''samples = pd.read_csv(config.SAMPLES_CSV)
print(len(samples), "sample")
samples["Label_Text"].value_counts().sort_index()''')

md("## 5. Warm-up (khong ghi ket qua)")
code('''pipeline.warmup(eb, samples, N_WARMUP)
tmp = os.path.join(config.OUT_DIR, "_warmup")
pipeline.run_sample(samples.iloc[0], eb, vlm, tmp, KB_MODE, KB_PARTS)   # nap kernel CUDA cua VLM
shutil.rmtree(tmp, ignore_errors=True)
meta = pipeline.write_run_meta(RUN_DIR, eb, vlm, KB_MODE, KB_PARTS, extra={"load": LOAD, "n_samples": len(samples)})
print("warm-up xong")''')

md("""## 6. Chay thu 5 sample → `output/test_5_sample` (**comment ca cell nay khi chay that**)

Chon 2 sample bat co + 3 sample co tat (nhan khac nhau), chay toan bo pipeline, roi kiem: VLM co trich dan dung
cach khong, bao cao co bi cat (cham gioi han token) khong, bo nho GPU dinh, phien ban thu vien.""")
code('''# ===== CHAY THU 5 SAMPLE — comment ca cell nay khi chay that =====
import torch, xgboost, sklearn, shap, transformers
TEST_DIR = os.path.join(config.OUT_DIR, "output", "test_5_sample")
os.makedirs(TEST_DIR, exist_ok=True)

row_of = {r["sample_id"]: r for _, r in samples.iterrows()}
flag_of = {}                                   # tra nhanh (khong anh, khong VLM) de biet sample nao bat co
for sid, r in row_of.items():
    ev = eb.build(r, out_dir=None, with_images=False)
    flag_of[sid] = bool(ev["alert"] and ev["lookup"]["flagged"])
pick = [s for s in row_of if flag_of[s]][:2]
used = {row_of[s]["Label_Text"] for s in pick}
for sid, r in row_of.items():
    if len(pick) == 5:
        break
    if not flag_of[sid] and r["Label_Text"] not in used:
        pick.append(sid); used.add(r["Label_Text"])
print(f"{sum(flag_of.values())}/{len(flag_of)} sample bat co; chay thu: {pick}")

pipeline.write_run_meta(TEST_DIR, eb, vlm, KB_MODE, KB_PARTS, extra={"test": True})
gpu = torch.cuda.is_available()
if gpu:
    torch.cuda.reset_peak_memory_stats()
tok = vlm.processor.tokenizer
for sid in pick:
    rec = pipeline.run_sample(row_of[sid], eb, vlm, TEST_DIR, KB_MODE, KB_PARTS)
    raw = open(os.path.join(TEST_DIR, sid, "output", "report.md"), encoding="utf-8").read() if rec.get("alert") else ""
    n_out = len(tok(raw).input_ids)
    print(f"\\n{sid}: flag={rec.get('flagged')} type={rec.get('attack_type')} sections={rec.get('n_sections')} "
          f"cites={rec.get('n_citations')} unknown={rec.get('n_unknown_citations')} "
          f"mitig_ok={rec.get('mitigations_valid')} out_tokens={n_out}/{config.VLM_MAX_NEW_TOKENS}"
          + ("  <-- CO THE BI CAT" if n_out >= config.VLM_MAX_NEW_TOKENS - 10 else "")
          + f" err={rec['run_error']} total={rec['latency_total']:.1f}s")
peak = f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if gpu else "khong co GPU"
print(f"\\nGPU peak: {peak} | xgboost {xgboost.__version__} | "
      f"scikit-learn {sklearn.__version__} | shap {shap.__version__} | transformers {transformers.__version__}")
res_t, lat_t = metrics.load_run(TEST_DIR)
display(metrics.summary(res_t)); display(metrics.latency_table(lat_t))
print(open(os.path.join(TEST_DIR, pick[0], "output", "report.md"), encoding="utf-8").read())''')

md("## 7. Chay ca lo")
code('''done = pipeline.done_ids(RUN_DIR)
todo = samples[~samples["sample_id"].isin(done)]
print(f"{len(done)} da xong, con {len(todo)}")
for i, (_, row) in enumerate(todo.iterrows(), 1):
    rec = pipeline.run_sample(row, eb, vlm, RUN_DIR, KB_MODE, KB_PARTS)
    print(f"[{i}/{len(todo)}] {rec['sample_id']:<34} type={rec.get('attack_type')} flag={rec.get('flagged')} "
          f"total={rec['latency_total']:.1f}s err={rec['run_error']}")''')

md("## 8. Tong hop")
code('''res, lat = metrics.load_run(RUN_DIR)
display(metrics.summary(res))
display(metrics.per_label(res))
display(metrics.latency_table(lat))''')

md("## 9. Dong goi ket qua")
code('''zip_path = shutil.make_archive(RUN_DIR, "zip", RUN_DIR)
print(zip_path, f"{os.path.getsize(zip_path) / 1e6:.1f} MB")''')

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                               "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlm_run.ipynb")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("->", out, len(C), "cell")
