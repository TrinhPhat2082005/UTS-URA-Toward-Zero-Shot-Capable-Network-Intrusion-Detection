"""Sinh run_layer2.ipynb. Chay mot lan: python _make_notebook.py"""
import json
import os

C = []


def _lines(s):
    """Chuoi nhieu dong -> list dong DUNG chuan nbformat: moi phan tu (tru
    phan tu cuoi) phai giu lai '\n' o cuoi. Thieu no thi cac tool doc notebook
    khac Jupyter (nbconvert, VSCode, GitHub preview) co the noi dong sai."""
    lines = s.rstrip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + (lines[-1:] if lines else [])


def md(s):
    C.append({"cell_type": "markdown", "metadata": {}, "source": _lines(s)})


def code(s):
    C.append({"cell_type": "code", "execution_count": None, "metadata": {},
              "outputs": [], "source": _lines(s)})


# --------------------------------------------------------------------------
md("""# Layer 2 - VLM forensic report (pilot, retriever L2 da chot)

Chay lai pilot (VLM tu chon 1 trong 14 nhan, 10 alert TP moi nhan) voi Layer 1 mono
va SHAP-KNN Retriever da chot: mot lan tinh L2 -> top-3 loai gan nhat + diem novelty
(trung binh L2 toi 10 lang gieng), co zero-day khi vuot tau_zd.

```
row -> Layer 1 (proba + TreeSHAP) -> retriever L2 (top-3 + novelty) -> 5 anh -> VLM -> report
```

Cac cell nap tuan tu. Chay tu tren xuong.

| Cell | Viec |
|---|---|
| 1-2 | Import, kiem tra duong dan |
| 3-5 | Nap Layer 1, retriever, VLM |
| 6-7 | Lay 10 alert TP moi nhan tu val_full (140 mau) |
| 8 | **Mot sample -> mot report** (xem ky truoc khi chay ca lo) |
| 9 | Chay ca lo |
| 10 | Bang so sanh voi pilot: theo loai, 14 va 12 loai, McNemar |
| 11-13 | Cac chi so con lai, confidence, latency |
| 14 | Do do nhay voi prompt (tuy chon) |""")

# --------------------------------------------------------------------------
md("""## 0. Setup tren Kaggle

**Local:** BO QUA cell nay.

**Kaggle:** code lay tu GitHub (private), data lay tu Kaggle Dataset.

Chuan bi mot lan:

1. **Token GitHub** — github.com/settings/tokens > Generate new token (classic)
   > tick `repo` > copy
2. **Luu vao Kaggle** — trong notebook: Add-ons > Secrets > Add a new secret,
   Label = `GITHUB_TOKEN`, Value = token, Save, roi tick **Attached**
3. **Gan dataset** — panel phai > Add Input > `layer2-data`

Roi bo comment cell duoi, sua `DATA` neu ten dataset khac, chay TRUOC moi cell.

| Bien | Y nghia |
|---|---|
| `UTS_DATA` | noi co `data/` + `artifacts/` (Kaggle dataset, read-only) |
| `UTS_OUT` | noi GHI ket qua — bat buoc tro ra `/kaggle/working` |""")

code('''# import os, sys, subprocess
# from kaggle_secrets import UserSecretsClient
#
# TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
# REPO  = f"https://{TOKEN}@github.com/TrinhPhat2082005/UTS-URA-Toward-Zero-Shot-Capable-Network-Intrusion-Detection.git"
# CODE  = "/kaggle/working/UTS-URA"
# DATA  = "/kaggle/input/layer2-data"      # sua neu ten dataset khac
#
# if not os.path.isdir(CODE):
#     subprocess.run(["git", "clone", "--depth", "1", REPO, CODE], check=True)
# else:
#     subprocess.run(["git", "-C", CODE, "pull"], check=True)
#
# LAYER2 = os.path.join(CODE, "LAYER2")
# os.environ["UTS_DATA"] = DATA
# os.environ["UTS_OUT"]  = "/kaggle/working/out"
# os.makedirs(os.environ["UTS_OUT"], exist_ok=True)
#
# for p in (LAYER2, os.path.join(LAYER2, "models")):
#     if p not in sys.path:
#         sys.path.insert(0, p)
# os.chdir(LAYER2)
#
# print("code:", LAYER2)
# print("data:", DATA)
# print("out :", os.environ["UTS_OUT"])
''')

md("""### Cai dependency

Kaggle da co torch/transformers/xgboost/shap. Hai goi duoi thuong thieu hoac
qua cu - ban bitsandbytes cu bao `ImportError: requires bitsandbytes>=0.46.1`
khi nap model 4-bit.

Restart session la mat het goi cai them, nen chay lai cell nay moi lan restart.
Cai xong bitsandbytes PHAI restart kernel (no nap thu vien CUDA luc import).""")

code('''!pip install -q -U "bitsandbytes>=0.46.1" qwen-vl-utils''')

md("""### Khong chac ten dataset?

Chay cell nay de xem Kaggle mount no thanh gi.""")

code('''# import os
# for d in sorted(os.listdir("/kaggle/input")):
#     print(d, "->", sorted(os.listdir(f"/kaggle/input/{d}"))[:6])''')

md("""### Sua code roi `git pull` lai?

Python giu module da import trong bo nho, nen phai reload sau khi pull.""")

code('''# import subprocess, importlib, config, prompts, pipeline, metrics
# subprocess.run(["git", "-C", "/kaggle/working/UTS-URA", "pull"], check=True)
# for m in (config, prompts, pipeline, metrics):
#     importlib.reload(m)
# config.describe()''')

md("## 1. Import")

code("""import os, sys, json, time, importlib
import numpy as np
import pandas as pd

# Cho phep import cac module canh notebook
HERE = os.getcwd()
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config, prompts, pipeline, metrics
for m in (config, prompts, pipeline, metrics):
    importlib.reload(m)

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)
print("import xong")""")

# --------------------------------------------------------------------------
md("""## 2. Kiem tra cau hinh

Dung lai ngay o day neu co file thieu, thay vi hong giua chung khi da nap VLM.""")

code("""ok = config.describe()
assert ok, "Thieu file - sua duong dan trong config.py truoc khi chay tiep."
""")

# --------------------------------------------------------------------------
md("""## 3. Layer 1

XGBoost gate + TreeExplainer. SHAP la gia tri chinh xac (khong lay mau), nen
vector nay dung lam khoa truy hoi duoc.""")

code("""l1 = pipeline.Layer1Engine(variant=config.VARIANT)""")

# --------------------------------------------------------------------------
md("""## 4. SHAP-KNN Retriever (L2)

Mot lan tinh khoang cach L2 tu vector SHAP toi moi vector cua R (moi flow attack
cua train_full, 14 nhan):

- top-3 = 3 loai co flow gan nhat (d_l nho nhat), dung mot quy tac cho ca 3 ung vien;
- novelty s(q) = trung binh L2 toi 10 lang gieng; co zero-day khi s(q) > tau_zd.

tau_zd doc tu `shap_pool_<variant>.meta.json` (khoa `novelty`). Neu dataset Kaggle
con pool cu (chua hieu chinh) thi tau_zd = None va co zero-day khong bao gio bat -
cell duoi se dung lai.""")

code("""ev = pipeline.EvidenceBuilder()
r = ev.retriever
print(f"R = {len(r.pool):,} vector, {len(r.labels)} nhan, k = {r.k}, tau_zd = {r.tau_zd}")
assert r.tau_zd is not None, "Pool tren dataset chua co tau_zd - upload lai shap_pool_*.meta.json moi"
assert len(r.labels) == 14, "Pool phai co du 14 nhan" """)

# --------------------------------------------------------------------------
md("""## 5. VLM

Nap Qwen2.5-VL 7B. Cell nay lau nhat (~vai phut) va ton VRAM nhat.

`do_sample=False` (greedy) de ket qua on dinh giua cac lan chay - dieu kien de
so sanh prompt cho co nghia.""")

code("""vlm = pipeline.VLMRunner()""")

# --------------------------------------------------------------------------
md("""## 6. Chay Layer 1 tren val_binary, tach TP/FP

Theo Paper 4.5: chay gate tren val_binary, giu lai cac dong NO DA BAO DONG,
roi tach theo ground truth:

- `tp` - Label = 1, gate bao dung
- `fp` - Label = 0, gate bao nham""")

code("""alerts, feats = pipeline.load_alert_pool(l1)
print(f"\\n{len(feats)} feature")
alerts["_set_type"].value_counts()""")

# --------------------------------------------------------------------------
md("""## 7. Lay mau — 10 alert moi loai tan cong

Chi lay nhom **tp** (alert dung). Nhanh danh gia false positive da bi bo: RAG
truy hoi bang chinh vector SHAP cua Layer 1 nen khong dung no de phu dinh
Layer 1 duoc (do thuc te: ca 5 FP deu khop ATTACK profile voi sim = 1.000).

10/nhan la muc cao nhat ma moi nhan deu dat (FTP_BruteForce chi co 10 alert)
-> du 14/14 nhan, tong 140 mau. Doi `per_class=2` de chay thu nhanh truoc.""")

code("""sample = pipeline.sample_stratified_tp(alerts, per_class=config.PER_CLASS)
sample["Label_Text"].value_counts()""")

# --------------------------------------------------------------------------
md("""## 8. MOT sample -> MOT report

Chay thu mot dong va in ca bao cao. Xem ky output nay truoc khi chay ca lo:
neu schema sai hoac anh hong thi sua o day, khong phai sau 100 luot.

`keep_images=True` de giu lai 5 anh ma xem.""")

code("""row = sample.iloc[0]

rec = pipeline.analyze_sample(
    row=row, l1=l1, ev=ev, vlm=vlm, feature_names=feats,
    prompt_variant="neutral",
    keep_images=True,
    meta={"row_idx": int(row["_row_idx"]),
          "set_type": row["_set_type"],
          "label": int(row["Label"]),
          "Label_Text": row["Label_Text"]},
)

print("run_error   :", rec.get("run_error"))
print("SU THAT     :", rec.get("Label_Text"), "| L1 proba", round(rec.get("xgb_proba", 0), 4))
print("RAG doan    :", rec.get("rag_predicted_label"),
      f"(novelty={rec.get('rag_novelty_score', 0):.3f}, zero_day={rec.get('rag_is_potential_zero_day')})")
print("VLM doan    :", rec.get("predicted_class"), "<-- M1 cham diem cai nay")
print("-" * 70)
print("severity    :", rec.get("severity"), "| MITRE:", rec.get("mitre_id"))
print("top_indicat :", rec.get("top_indicators"))
print("SHAP top-5  :", [f["name"] for f in (rec.get("shap_top_features") or [])])
print("conf logprob:", rec.get("confidence_logprob"),
      f"(token do: {rec.get('conf_token')!r})", "| tu khai:", rec.get("confidence_stated"))
print("n_sections  :", rec.get("n_sections"), "/6 | parse_error:", rec.get("parse_error"))
print("latency     :", round(rec.get("latency_s", 0), 1), "s")""")

code("""# Bao cao day du
print(rec.get("raw_output") or rec.get("traceback"))""")

code("""# Khoi RAG context da dua vao prompt
print(rec.get("rag_context"))""")

# --------------------------------------------------------------------------
md("""## 9. Chay ca lo

Ghi tung dong ngay khi xong (`append_jsonl`) - dut giua chung khong mat phan
da chay. Chay lai cell nay se GHI TIEP vao file cu; xoa file truoc neu muon
lam lai tu dau.""")

code("""PROMPT_VARIANT = "neutral"
# Ten file rieng cho lan chay voi retriever L2, khong ghi de len ket qua pilot cu
OUT = os.path.join(config.OUT_DIR, f"results_{config.VARIANT}_{PROMPT_VARIANT}_l2rag_v3.jsonl")

# Bo qua cac sample da chay xong (cho phep chay tiep sau khi dut)
done = set()
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as f:
        for ln in f:
            try:
                r = json.loads(ln)
                done.add((r.get("set_type"), r.get("row_idx")))
            except Exception:
                pass
    print(f"da co {len(done)} ban ghi - se bo qua")

t_start = time.time()
for i, (_, row) in enumerate(sample.iterrows(), 1):
    key = (row["_set_type"], int(row["_row_idx"]))
    if key in done:
        continue

    rec = pipeline.analyze_sample(
        row=row, l1=l1, ev=ev, vlm=vlm, feature_names=feats,
        prompt_variant=PROMPT_VARIANT,
        keep_images=False,
        meta={"row_idx": int(row["_row_idx"]),
              "set_type": row["_set_type"],
              "label": int(row["Label"]),
              "Label_Text": row["Label_Text"],
              "variant": config.VARIANT},
    )
    pipeline.append_jsonl(rec, OUT)

    el = time.time() - t_start
    print(f"[{i:3d}/{len(sample)}] {rec['set_type']:>2} "
          f"{str(rec.get('Label_Text'))[:20]:<20} -> {str(rec.get('predicted_class'))[:20]:<20} "
          f"{rec.get('latency_s', 0):5.1f}s  (da chay {el/60:.1f} phut)")

print(f"\\nXong trong {(time.time()-t_start)/60:.1f} phut -> {OUT}")""")

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
md("""## 10. Cham diem — G1 den G4

Xem METRIC.md de biet dinh nghia day du. Tom tat:

| | Do gi |
|---|---|
| **G1** M1 accuracy, M2 per-class, M3 MITRE | VLM co xac dinh dung loai tan cong? |
| **G2** M4 adherence, M5 harm, **M6 VLM vs RAG**, M7 schema | Dua bang chung hay tu bia? |
| **G3** M8 grounding, M9 halluc, M10 convergence | Report dung duoc? |
| **G4** M11 latency | Real-time? |

**M6 la chi so ban chinh** — so accuracy cua VLM voi accuracy cua chinh RAG:
VLM cong them, chep lai, hay lam hong ket qua truy hoi.""")

code("""df = metrics.load_results(OUT)
metrics.report_all(df)""")

md("""### So sanh voi pilot cu (cung cach doc nhu bang theo loai)

Dung theo nhan goc CIC-IDS2018. "VLM pha RAG" = retriever dung, VLM sai.
Hai loai song sinh (FTP-BruteForce, DoS-SlowHTTPTest) la tan cong that bai,
flow trung nhau [Liu 2022] -> bao cao rieng, so chinh la 12 loai.""")

code("""from collections import defaultdict
from scipy.stats import binomtest

TW = {"FTP_BruteForce", "DoS_attacks_SlowHTTPTest"}
ok = df[df["run_error"].isna()]
per = defaultdict(lambda: [0, 0, 0, 0, 0])          # n, rag dung, vlm dung, vlm pha rag, vlm sua rag
for _, r in ok.iterrows():
    y = r["Label_Text"]
    a = metrics._match(r.get("rag_predicted_label") or "", y)
    b = metrics._match(r.get("predicted_class") or "", y)
    p = per[y]; p[0] += 1; p[1] += a; p[2] += b; p[3] += a and not b; p[4] += b and not a

rows = [{"loai": y + (" (song sinh)" if y in TW else ""), "n": n, "RAG dung": a, "VLM dung": b,
         "VLM pha RAG": c, "VLM sua RAG": d} for y, (n, a, b, c, d) in sorted(per.items())]
for name, keep in (("14 loai", lambda y: True), ("12 loai (bo song sinh)", lambda y: y not in TW)):
    ks = [y for y in per if keep(y)]
    s = [sum(per[y][i] for y in ks) for i in range(5)]
    p_mc = binomtest(s[3], s[3] + s[4], 0.5).pvalue if s[3] + s[4] else float("nan")
    rows.append({"loai": name, "n": s[0], "RAG dung": f"{s[1]} ({s[1]/s[0]:.1%})",
                 "VLM dung": f"{s[2]} ({s[2]/s[0]:.1%})", "VLM pha RAG": s[3],
                 "VLM sua RAG": f"{s[4]}  (McNemar exact p = {p_mc:.2g})"})
display(pd.DataFrame(rows))

# Zero-day: so alert bi gan co (alert TP cua loai DA BIET -> ky vong ~5%)
print(f"co zero-day: {ok['rag_is_potential_zero_day'].mean():.1%} cua {len(ok)} alert")

# Dau ra: dung / UNKNOWN / sai nhung la ten lop hop le / khong hop le, tach co / khong co.
# Moi alert o day thuoc loai DA BIET nen UNKNOWN tinh la sai (tu choi nham).
CLASSES = sorted(TW | {"Bot", "Brute_Force_Web", "Brute_Force_XSS", "DDOS_attack_HOIC", "DDOS_attack_LOIC_UDP",
                       "DDoS_attacks_LOIC_HTTP", "DoS_attacks_GoldenEye", "DoS_attacks_Hulk",
                       "DoS_attacks_Slowloris", "Infiltration", "SQL_Injection", "SSH_Bruteforce"})
def kind(r):
    v, y = (r.get("predicted_class") or "").strip(), r["Label_Text"]
    if metrics._match(v, y):
        return "dung"
    if v.upper().startswith("UNKNOWN"):
        return "UNKNOWN"
    return "sai (lop hop le)" if any(metrics._match(v, c) for c in CLASSES) else "khong hop le"
ok = ok.assign(kind=ok.apply(kind, axis=1),
               nhom=ok["rag_is_potential_zero_day"].map({True: "co co", False: "khong co"}))
display(pd.crosstab(ok["nhom"], ok["kind"], margins=True))
print("Output khong hop le:", ok.loc[ok["kind"] == "khong hop le", "predicted_class"].tolist())""")

# --------------------------------------------------------------------------
md("## 11. Tung nhom rieng (neu can xem ky)")

code("""m1 = metrics.attack_id_metrics(df)
pc = metrics.per_class(df)
display(pc)""")

code("""g2 = metrics.grounding_metrics(df)
g3 = metrics.report_quality(df)""")

# --------------------------------------------------------------------------
md("""## 12. Confidence

So hai nguon: `confidence_logprob` (xac suat token) va `confidence_stated`
(so model tu khai). Run cu cho thay cot tu khai gan nhu hang so 0.9 nen khong
quet nguong duoc — kiem tra lai o day.""")

code("""for col in ["confidence_logprob", "confidence_stated"]:
    metrics.confidence_auroc(df, col)
    sw = metrics.confidence_sweep(df, col)
    if len(sw):
        display(sw)""")

# --------------------------------------------------------------------------
md("## 13. Latency + chat luong output")

code("""display(metrics.latency_breakdown(df))

print("predicted_class:", df["predicted_class"].value_counts(dropna=False).head(10).to_dict())
print("parse_error    :", df["parse_error"].notna().sum(), "/", len(df))
print("run_error      :", df["run_error"].notna().sum(), "/", len(df))
print("zero-day flag  :", df["rag_is_potential_zero_day"].sum(), "/", len(df))""")

code("""# Ca VLM doan SAI - doc de hieu kieu loi
d = metrics.usable(df)
bad = d[~d.apply(lambda r: metrics._match(r["predicted_class"], r.get("Label_Text")), axis=1)]
print(f"{len(bad)}/{len(d)} ca sai")
display(bad[["Label_Text", "rag_predicted_label", "predicted_class",
             "rag_novelty_score", "confidence_logprob"]].head(20))""")

# --------------------------------------------------------------------------
md("""## 14. Do do nhay voi prompt (tuy chon)

Chay cung mot bo mau qua 3 prompt variant de tach: ket qua den tu **bang chung**
hay tu **cach dat cau hoi**?

- `neutral` — khong uu tien nguon nao
- `trust_rag` — noi RAG dang tin
- `trust_visual` — noi RAG co the sai, doc chart truoc

Run cu (v6/v7/v13, cung 40 mau) cho 37/40 doi ket qua chi vi doi chu trong
prompt. Neu con lap lai o day thi do la ket qua phai bao cao.

Ton them 1 luot VLM moi sample moi variant.""")

code("""RUN_SENSITIVITY = False   # dat True de chay

if RUN_SENSITIVITY:
    sub = pipeline.sample_stratified_tp(alerts, per_class=3)
    OUT_S = os.path.join(config.OUT_DIR, f"results_sensitivity_{config.VARIANT}.jsonl")

    for variant in ["neutral", "trust_rag", "trust_visual"]:
        print(f"\\n--- {variant} ---")
        for _, row in sub.iterrows():
            rec = pipeline.analyze_sample(
                row=row, l1=l1, ev=ev, vlm=vlm, feature_names=feats,
                prompt_variant=variant, keep_images=False,
                meta={"row_idx": int(row["_row_idx"]),
                      "set_type": row["_set_type"],
                      "label": int(row["Label"]),
                      "Label_Text": row["Label_Text"]},
            )
            pipeline.append_jsonl(rec, OUT_S)
            print(f"  {row['Label_Text']:<24} -> {rec.get('predicted_class')}")

    dfs = metrics.load_results(OUT_S)
    per, piv = metrics.prompt_sensitivity(dfs)
    display(per); display(piv)""")
# --------------------------------------------------------------------------
nb = {
    "cells": C,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_layer2.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"wrote {out} ({len(C)} cells)")
