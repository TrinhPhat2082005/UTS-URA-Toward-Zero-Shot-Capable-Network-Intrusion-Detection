"""
Duong dan cho danh gia Layer 1 (RQ1, RQ2 — Protocol A).

Code (repo UTS_URA) va du lieu/checkpoint (workspace) nam o hai noi, nen moi duong dan doc
tu bien moi truong; mac dinh coi repo nam trong workspace (workspace/UTS_URA):

  UTS_WS            thu muc goc workspace                (mac dinh: cha cua repo)
  UTS_BENCH         zeroshot_benchmark/ (split + checkpoint Protocol A)
  UTS_LAYER1_CODE   noi chua models/ va common/ cua Layer 1 (mac dinh: LAYER1/03_training cua repo)
  UTS_LAYER1_ART    checkpoint cua paper, 14 fold LOAO
  UTS_BASELINE_ART  checkpoint cua baseline study (teammate)
  UTS_OPTIFOREST    thu muc chua package detectors/ cua OptIForest (mac dinh: <cha workspace>/isolation_forest)
  UTS_L1_RESULTS    noi ghi ket qua                       (mac dinh: LAYER1/output)
"""
import os
import sys
from pathlib import Path

LAYER1 = Path(__file__).resolve().parents[2]
REPO = LAYER1.parent
WS = Path(os.environ.get("UTS_WS", REPO.parent))
BENCH = Path(os.environ.get("UTS_BENCH", WS / "zeroshot_benchmark"))
SPLITS_ROOT = BENCH / "03_splits_v2"            # train_benign_only.csv + by_label/
SPLITS = SPLITS_ROOT / "by_label"               # 14 fold LOAO, moi fold mot thu muc
L1_CODE = Path(os.environ.get("UTS_LAYER1_CODE", LAYER1 / "03_training"))
L1_ART = Path(os.environ.get("UTS_LAYER1_ART", BENCH / "04_training_v2" / "artifacts"))   # ten thu muc du lieu
BASE_ART = Path(os.environ.get("UTS_BASELINE_ART", WS / "baselines_friends" / "artifacts"))
OPTIFOREST_CODE = Path(os.environ.get("UTS_OPTIFOREST", WS.parent / "isolation_forest"))

OUT = Path(os.environ.get("UTS_L1_RESULTS", LAYER1 / "output"))
OUT_SCORES = OUT / "scores"                     # cache diem: paper.npz, baseline_<model>.npz
OUT_TABLES = OUT / "tables"                     # bang, hinh, verify.md, latency.json


def use_layer1_code():
    """Cho phep `from models... import` cua Layer 1 (autoencoder, svdd, clad)."""
    if str(L1_CODE) not in sys.path:
        sys.path.insert(0, str(L1_CODE))


def ensure(d):
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d)


def rel(p):
    """Duong dan de in ra: tuong doi so voi LAYER1 neu nam trong do (output notebook khong chua duong dan tuyet doi)."""
    try:
        return Path(p).resolve().relative_to(LAYER1).as_posix()
    except ValueError:
        return str(p)
