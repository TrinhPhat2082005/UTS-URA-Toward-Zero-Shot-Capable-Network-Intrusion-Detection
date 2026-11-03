"""
Duong dan dung chung cho moi script thi nghiem.

Code (repo UTS_URA) va du lieu/checkpoint (workspace) co the nam o hai noi, nen moi
duong dan doc tu bien moi truong; mac dinh theo cau truc workspace:

  UTS_WS           thu muc goc workspace                (mac dinh: cha cua LAYER2/)
  UTS_BENCH        zeroshot_benchmark/ (split + checkpoint Protocol A)
  UTS_LAYER1_CODE  noi chua models/ va common/ cua Layer 1 (04_training_v2 hoac UTS_URA/LAYER1/03_training)
  UTS_LAYER1_ART   checkpoint Layer 1 cua 14 fold LOAO
  UTS_RESULTS      noi ghi ket qua thi nghiem            (mac dinh: LAYER2/output)
  UTS_DATA         du lieu Protocol B cua LAYER2 (doc trong config.py)
  UTS_BASELINE_ART checkpoint cac baseline Layer 1 mo rong (baseline study)
  UTS_L1_RESULTS   ket qua thi nghiem Layer 1 (mac dinh: LAYER1/output cua repo)
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
OUT_RQ3A = OUT / "rq3a_identify"                # tab:res-ident
OUT_RQ3B = OUT / "rq3b_novelty"                 # tab:res-novelty, end-to-end zero-day
OUT_RQ5 = OUT / "rq5_latency"                   # tab:res-latency, cot latency cua tab:res-known

# Thi nghiem Layer 1 (RQ1, RQ2) nam o LAYER1/04_experiments, ket qua o LAYER1/output cua repo.
# Chay tu repo thi LAYER1 la anh em cua LAYER2; ban sao workspace/LAYER2 tim repo o workspace/UTS_URA.
L1_REPO = next((r / "LAYER1" for r in (LAYER2.parent, WS / "UTS_URA") if (r / "LAYER1" / "04_experiments").is_dir()),
               LAYER2.parent / "LAYER1")
L1_EXP = L1_REPO / "04_experiments"             # package layer1_eval
L1_OUT = Path(os.environ.get("UTS_L1_RESULTS", L1_REPO / "output"))
L1_SCORES = L1_OUT / "scores"                   # paper.npz (6 model paper), baseline_<model>.npz — RQ3b doc lai

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
