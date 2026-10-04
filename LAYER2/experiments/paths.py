"""
Duong dan dung chung cho moi script thi nghiem.

Code (repo UTS_URA) va du lieu/checkpoint (workspace) co the nam o hai noi, nen moi
duong dan doc tu bien moi truong; mac dinh theo cau truc workspace:

  UTS_WS           thu muc goc workspace                (mac dinh: cha cua LAYER2/)
  UTS_BENCH        zeroshot_benchmark/ (split + checkpoint Protocol A)
  UTS_LAYER1_CODE  noi chua models/ va common/ cua Layer 1 (04_training_v2 hoac UTS_URA/LAYER1)
  UTS_LAYER1_ART   checkpoint Layer 1 cua 14 fold LOAO
  UTS_RESULTS      noi ghi ket qua thi nghiem            (mac dinh: LAYER2/output)
  UTS_DATA         du lieu Protocol B cua LAYER2 (doc trong config.py)
  UTS_BASELINE_ART checkpoint cac baseline Layer 1 mo rong (rq1_rq2_baselines/)
"""
import os
import sys
from pathlib import Path

LAYER2 = Path(__file__).resolve().parents[1]
WS = Path(os.environ.get("UTS_WS", LAYER2.parent))
BENCH = Path(os.environ.get("UTS_BENCH", WS / "zeroshot_benchmark"))
SPLITS_ROOT = BENCH / "03_splits_v2"            # Protocol A: train_benign_only.csv + by_label/
SPLITS = SPLITS_ROOT / "by_label"               # 14 fold LOAO, moi fold mot thu muc
L1_CODE = Path(os.environ.get("UTS_LAYER1_CODE", BENCH / "04_training_v2"))
L1_ART = Path(os.environ.get("UTS_LAYER1_ART", BENCH / "04_training_v2" / "artifacts"))
BASE_ART = Path(os.environ.get("UTS_BASELINE_ART", WS / "baselines_friends" / "artifacts"))
PROFILES = LAYER2 / "knowledge_base" / "attack_profiles"
ARTIFACTS = Path(os.environ.get("UTS_DATA", LAYER2)) / "artifacts"   # giong config.ARTIFACTS_DIR
LOAO_POOLS = ARTIFACTS / "loao_pools"           # R cua tung fold Protocol A

OUT = Path(os.environ.get("UTS_RESULTS", LAYER2 / "output"))
OUT_L1 = OUT / "rq1_rq2_layer1"                 # tab:res-known, tab:res-zs, tab:res-zs-type
OUT_BASE = OUT / "rq1_rq2_baselines"            # bang baseline mo rong, cung quy tac nguong
OUT_RQ3A = OUT / "rq3a_identify"                # tab:res-ident
OUT_RQ3B = OUT / "rq3b_novelty"                 # tab:res-novelty, end-to-end zero-day
OUT_RQ5 = OUT / "rq5_latency"                   # tab:res-latency, cot latency cua tab:res-known

# Bien the Layer 1 duoc bao cao trong paper (ten thu muc checkpoint trong L1_ART)
SUPERVISED = ["xgboost_only", "ae_xgboost_mono", "svdd_xgboost_mono"]
ONE_CLASS = ["ae_only", "svdd_only"]
TWINS = {"FTP_BruteForce", "DoS_attacks_SlowHTTPTest"}   # loai bo khoi macro 12 loai [Liu2022]


def use_layer1_code():
    """Cho phep `from models... import` / `from common... import` cua Layer 1."""
    if str(L1_CODE) not in sys.path:
        sys.path.insert(0, str(L1_CODE))


def use_layer2_code():
    """Cho phep `import config, pipeline, shap_knn_retriever` cua LAYER2."""
    if str(LAYER2) not in sys.path:
        sys.path.insert(0, str(LAYER2))


def ensure(d):
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d)
