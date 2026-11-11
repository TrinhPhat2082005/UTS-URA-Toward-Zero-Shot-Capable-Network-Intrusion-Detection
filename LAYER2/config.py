"""
Cau hinh duong dan + tham so cho Layer 2.

Moi duong dan deu suy ra tu PROJECT_ROOT nen notebook chay duoc ca o local
lan Kaggle (chi can doi PROJECT_ROOT hoac dat bien moi truong UTS_ROOT).

Khong co logic o day - chi hang so. Muc dich: notebook khong hardcode path.
"""
import os

# --------------------------------------------------------------------------
# Goc du an
# --------------------------------------------------------------------------
# Thu muc chua chinh config.py nay. Suy truc tiep tu __file__ (khong ghep chuoi
# "LAYER2" vao PROJECT_ROOT nua) de khong phu thuoc ten thu muc: tren Kaggle,
# repo clone ve co the nam o bat ky dau, va Linux phan biet hoa/thuong.
LAYER2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get("UTS_ROOT", os.path.dirname(LAYER2_DIR))

# Noi chua DU LIEU NANG (csv, checkpoint, SHAP pool).
#
# Code nam tren GitHub, data nam tren Kaggle Dataset - hai cho khac nhau. Dat
# UTS_DATA tro toi thu muc dataset da mount, vi du:
#     os.environ["UTS_DATA"] = "/kaggle/input/layer2-data"
# Khong dat thi mac dinh nam ngay trong LAYER2 (dung cho may local).
DATA_ROOT = os.environ.get("UTS_DATA", LAYER2_DIR)

# --------------------------------------------------------------------------
# Layer 1: model + scaler
#
# Ba variant deu dung chung interface (predict_proba tren 78 feature da scale).
# Doi VARIANT la doi ca model, scaler va build pool SHAP tuong ung.
# --------------------------------------------------------------------------
VARIANT = "ae_xgboost"            # "ae_xgboost" | "svdd_xgboost"

# Checkpoint train boi LAYER1/03_training/train_protocol_b.py tren <UTS_DATA>/data/train_full.csv.
# Moi thu muc co: model.pkl, scaler.pkl, feature_names.json, metrics.json
# va encoder tuong ung (ae_model.pt hoac svdd/).
_L1 = os.path.join(DATA_ROOT, "artifacts", "layer1")

# Code Layer 1 (kien truc AE/SVDD trong models/): LAYER1/03_training cua repo. LAYER2 khong giu
# ban sao nao; gate/engine.py import tu day. Dat UTS_LAYER1_CODE neu nam cho khac.
LAYER1_CODE_DIR = os.environ.get("UTS_LAYER1_CODE") or next(
    (p for p in (os.path.join(PROJECT_ROOT, "LAYER1", "03_training"),
                 os.path.join(PROJECT_ROOT, "UTS_URA", "LAYER1", "03_training"))
     if os.path.isdir(p)),
    os.path.join(PROJECT_ROOT, "LAYER1", "03_training"))


def use_layer1_code():
    """Cho phep `from models.autoencoder import ...` (kien truc Layer 1 cua LAYER1)."""
    import sys
    if LAYER1_CODE_DIR not in sys.path:
        sys.path.insert(0, LAYER1_CODE_DIR)

LAYER1_ARTIFACTS = {
    "ae_xgboost": {
        "model":    os.path.join(_L1, "ae_xgboost", "model.pkl"),
        "scaler":   os.path.join(_L1, "ae_xgboost", "scaler.pkl"),
        "encoder":  os.path.join(_L1, "ae_xgboost", "ae_model.pt"),
        "features": os.path.join(_L1, "ae_xgboost", "feature_names.json"),
        "metrics":  os.path.join(_L1, "ae_xgboost", "metrics.json"),
    },
    "svdd_xgboost": {
        "model":    os.path.join(_L1, "svdd_xgboost", "model.pkl"),
        "scaler":   os.path.join(_L1, "svdd_xgboost", "scaler.pkl"),
        "encoder":  os.path.join(_L1, "svdd_xgboost", "svdd"),
        "features": os.path.join(_L1, "svdd_xgboost", "feature_names.json"),
        "metrics":  os.path.join(_L1, "svdd_xgboost", "metrics.json"),
    },
}

# Gia tri clip cua RobustScaler - phai khop LAYER1/03_training/train_protocol_b.py.
CLIP_VAL = 20.0

# Nguong Layer 1: None = doc "threshold" tu metrics.json, la nguong theo ngan sach
# FPR alpha = 0.5% tren benign cua TRAIN (LAYER1/03_training/train_protocol_b.py, OPERATING_ALPHA;
# metrics.json con luu thresholds_by_alpha cho cac alpha khac). Khong tune tren val, vi val la
# tap bao cao TP/FP cho Layer 2. Dat mot so cu the de ghi de.
LAYER1_THRESHOLD = None

# --------------------------------------------------------------------------
# Du lieu
#
# Paper 4.5: chay Layer 1 tren val_binary, tach alert thanh TP/FP theo ground
# truth, lay mau bang nhau moi nhom.
# --------------------------------------------------------------------------
DATA_DIR = os.path.join(DATA_ROOT, "data")
TRAIN_FULL_CSV = os.path.join(DATA_DIR, "train_full.csv")
VAL_BINARY_CSV = os.path.join(DATA_DIR, "val_full.csv")

# Cot metadata trong CSV - KHONG phai feature dau vao model.
#
# Label_Text la ten nhan goc (Benign / 14 loai tan cong), dung de lay mau va
# doi chieu, khong phai feature.
#
# Timestamp / Flow ID / Src IP / Src Port / Dst IP chi co trong cac split cu;
# giu lai trong danh sach de config nay con doc duoc ca hai dinh dang.
META_COLS = ["Label", "Label_Text", "Flow ID", "Src IP", "Src Port", "Dst IP",
             "Timestamp"]

# --------------------------------------------------------------------------
# RAG: SHAP k-NN
# --------------------------------------------------------------------------
# Tap tham chieu R (retrieval/reference_set.py):
#   reference_sets/protocol_b/<variant>.npz          Protocol B (pipeline trien khai)
#   reference_sets/protocol_a/<fold>/<variant>.npz   Protocol A, tung fold LOAO (R3, Setup 2)
REFERENCE_SETS_DIR = os.path.join(DATA_ROOT, "artifacts", "reference_sets")
REFERENCE_SET_NPZ = os.path.join(REFERENCE_SETS_DIR, "protocol_b", f"{VARIANT}.npz")
ATTACK_PROFILES_DIR = os.path.join(
    LAYER2_DIR, "knowledge_base", "attack_profiles"
)

RAG_TOP_K = 3                     # 3 loai co khoang cach L2 toi flow gan nhat nho nhat
RAG_NOVELTY_K = 10                # s(q) = trung binh L2 toi 10 lang gieng gan nhat
# tau_zd: None -> doc tu <tap R>.meta.json ("novelty" -> "tau_zd"), hieu chinh boi
# experiments/rq3_retrieval/02_calibrate_novelty.py (Q_0.95 tren alert da biet).
RAG_ZERODAY_THRESHOLD = None

# Khong loai nhan nao: tap Layer 2 co du 14 loai tan cong, Infiltration gio la
# nhan binh thuong chu khong con bi giu lam zero-shot.
RAG_EXCLUDE_LABELS = []

# --------------------------------------------------------------------------
# VLM
# --------------------------------------------------------------------------
VLM_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_LOAD_4BIT = True
VLM_MAX_NEW_TOKENS = 768

# Gioi han vision token moi anh (don vi: so patch 28x28).
#
# Bat buoc phai dat: Qwen2.5-VL chia anh thanh patch theo kich thuoc that, nen
# 5 anh o 150-200 dpi lam mot lan allocate doi ~6.9 GiB -> OOM tren GPU 16GB.
# 768 patch ~ 602K pixel/anh (vd 900x670), du doc chu trong bieu do.
# Neu van OOM: ha VLM_MAX_PIXELS xuong 512 * 28 * 28.
VLM_MIN_PIXELS = 256 * 28 * 28
VLM_MAX_PIXELS = 768 * 28 * 28
VLM_TEMPERATURE = 0.0             # greedy: bo nhieu tu sampling khi do lai verdict

# Bat logprob de lay confidence tu model thay vi de model tu khai.
# Ly do: do lai cac run cu (v6/v7/v13) cho thay confidence tu khai gan nhu luon
# = 0.9 bat ke dung sai, nen khong quet duoc threshold nao ca.
VLM_RETURN_LOGPROBS = True

# --------------------------------------------------------------------------
# Lay mau danh gia
# --------------------------------------------------------------------------
# 10 alert moi loai tan cong -> 140 mau, du 14/14 nhan.
# 10 la muc cao nhat moi nhan deu dat (FTP_BruteForce chi co 10 alert).
# Chi lay nhom tp: nhanh danh gia false positive da bo (xem METRIC.md).
PER_CLASS = 10
SAMPLE_SEED = 42

N_PER_GROUP = PER_CLASS           # giu lai cho sample_balanced() cu

# --------------------------------------------------------------------------
# Dau ra
# --------------------------------------------------------------------------
ARTIFACTS_DIR = os.path.join(DATA_ROOT, "artifacts")

# Noi GHI ket qua. Tren Kaggle, LAYER2 nam o /kaggle/input/... (read-only) nen
# phai tro OUT_DIR sang /kaggle/working: dat bien moi truong UTS_OUT truoc khi
# import config.
OUT_DIR = os.environ.get("UTS_OUT", ARTIFACTS_DIR)
EVIDENCE_DIR = os.path.join(OUT_DIR, "evidence_cache")
RESULTS_JSONL = os.path.join(OUT_DIR, "results.jsonl")


def layer1_threshold(variant=None):
    """Nguong quyet dinh cua Layer 1.

    Uu tien LAYER1_THRESHOLD neu duoc dat tay; neu khong thi doc "threshold" tu
    metrics.json cua variant (nguong theo ngan sach FPR tren benign cua TRAIN).
    """
    if LAYER1_THRESHOLD is not None:
        return float(LAYER1_THRESHOLD)
    import json
    p = LAYER1_ARTIFACTS[variant or VARIANT]["metrics"]
    with open(p, encoding="utf-8") as f:
        return float(json.load(f)["threshold"])


def describe():
    """In cau hinh dang dung + canh bao file thieu. Goi o cell dau notebook."""
    art = LAYER1_ARTIFACTS[VARIANT]
    rows = [
        ("code (LAYER2)", LAYER2_DIR, os.path.isdir(LAYER2_DIR)),
        ("data (UTS_DATA)", DATA_ROOT, os.path.isdir(DATA_ROOT)),
        ("variant", VARIANT, True),
        ("L1 model", art["model"], os.path.exists(art["model"])),
        ("L1 scaler", art["scaler"], os.path.exists(art["scaler"])),
        ("L1 encoder", art["encoder"], os.path.exists(art["encoder"])),
        ("train_full", TRAIN_FULL_CSV, os.path.exists(TRAIN_FULL_CSV)),
        ("val_full", VAL_BINARY_CSV, os.path.exists(VAL_BINARY_CSV)),
        ("reference R", REFERENCE_SET_NPZ, os.path.exists(REFERENCE_SET_NPZ)),
        ("profiles", ATTACK_PROFILES_DIR, os.path.isdir(ATTACK_PROFILES_DIR)),
    ]
    missing = []
    for name, val, ok in rows:
        mark = "ok " if ok else "MISSING"
        short = val if len(str(val)) < 70 else "..." + str(val)[-67:]
        print(f"  [{mark}] {name:<12} {short}")
        if not ok:
            missing.append(name)
    try:
        thr = f"{layer1_threshold():.4f}" + (" (tu metrics.json)" if LAYER1_THRESHOLD is None else "")
    except Exception:
        thr = "chua co metrics.json"
    print(f"\n  threshold={thr}  top_k={RAG_TOP_K}  "
          f"zero_day_thr={RAG_ZERODAY_THRESHOLD if RAG_ZERODAY_THRESHOLD is not None else 'doc tu meta.json cua tap R'}")
    print(f"  mau: {PER_CLASS}/nhan x 14 nhan = {PER_CLASS * 14}  seed={SAMPLE_SEED}")
    print(f"  ghi ket qua vao: {OUT_DIR}")
    if missing:
        print(f"\n  !! thieu: {', '.join(missing)}")
    return not missing
