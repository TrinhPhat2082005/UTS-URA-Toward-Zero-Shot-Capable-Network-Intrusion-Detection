"""
Duong dan dung chung cho thi nghiem Layer 1 (RQ1, RQ2 — Protocol A).

Code (repo UTS_URA) va du lieu/checkpoint (workspace) nam o hai noi, nen moi duong dan doc
tu bien moi truong; mac dinh coi repo nam trong workspace (workspace/UTS_URA):

  UTS_WS           thu muc goc workspace                (mac dinh: cha cua repo)
  UTS_BENCH        zeroshot_benchmark/ (split + checkpoint Protocol A)
  UTS_LAYER1_CODE  noi chua models/ va common/ cua Layer 1 (mac dinh: LAYER1/04_training_v2 cua repo)
  UTS_LAYER1_ART   checkpoint Layer 1 cua 14 fold LOAO
  UTS_BASELINE_ART checkpoint cac baseline Layer 1 mo rong (rq1_rq2_baselines/)
  UTS_L1_RESULTS   noi ghi ket qua thi nghiem Layer 1  (mac dinh: LAYER1/output)

RQ3b cua LAYER2 doc lai ket qua o day (diem benign train, scores_svdd_xgboost.npz) qua
LAYER2/experiments/paths.py (OUT_L1, OUT_BASE).
"""
import os
import sys
from pathlib import Path

LAYER1 = Path(__file__).resolve().parents[1]
REPO = LAYER1.parent
WS = Path(os.environ.get("UTS_WS", REPO.parent))
BENCH = Path(os.environ.get("UTS_BENCH", WS / "zeroshot_benchmark"))
SPLITS_ROOT = BENCH / "03_splits_v2"            # Protocol A: train_benign_only.csv + by_label/
SPLITS = SPLITS_ROOT / "by_label"               # 14 fold LOAO, moi fold mot thu muc
L1_CODE = Path(os.environ.get("UTS_LAYER1_CODE", LAYER1 / "04_training_v2"))
L1_ART = Path(os.environ.get("UTS_LAYER1_ART", BENCH / "04_training_v2" / "artifacts"))
BASE_ART = Path(os.environ.get("UTS_BASELINE_ART", WS / "baselines_friends" / "artifacts"))

OUT = Path(os.environ.get("UTS_L1_RESULTS", LAYER1 / "output"))
OUT_L1 = OUT / "rq1_rq2_layer1"                 # tab:res-known, tab:res-zs, tab:res-zs-type
OUT_BASE = OUT / "rq1_rq2_baselines"            # bang baseline mo rong, cung quy tac nguong

# Bien the Layer 1 duoc bao cao trong paper (ten thu muc checkpoint trong L1_ART)
SUPERVISED = ["xgboost_only", "ae_xgboost_mono", "svdd_xgboost_mono"]
ONE_CLASS = ["ae_only", "svdd_only"]


def use_layer1_code():
    """Cho phep `from models... import` / `from common... import` cua Layer 1."""
    if str(L1_CODE) not in sys.path:
        sys.path.insert(0, str(L1_CODE))


def ensure(d):
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d)
