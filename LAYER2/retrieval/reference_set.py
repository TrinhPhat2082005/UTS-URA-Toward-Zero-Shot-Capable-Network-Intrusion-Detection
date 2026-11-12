"""
Tap tham chieu R (paper Section 3.5): vector SHAP cua moi flow attack cua tap train, gom theo loai.

    <UTS_DATA>/artifacts/reference_sets/protocol_b/<variant>.npz          Protocol B (train_full.csv), file nay build
    <UTS_DATA>/artifacts/reference_sets/protocol_a/<fold>/<variant>.npz   Protocol A, tung fold LOAO
                                                                          (experiments/rq3_retrieval/01_build_loao_pools.py)

Moi file, cho moi loai <L> (cung thu tu dong trong ca ba khoa):
    label__<L>  (n, d) float32   vector SHAP
    x__<L>      (n, 78) float64  gia tri tho cua 78 feature goc (bang doi chieu C, KB)      [tu 2026-11-13]
    row__<L>    (n,) int64       chi so dong trong train_full.csv (truy lai flow goc)       [tu 2026-11-13]
va feature_names, base_feature_names; kem <file>.meta.json (nguon, so vector moi loai, ... va
"novelty" -> "tau_zd" sau khi chay 02_calibrate_novelty.py).
Vector SHAP tinh bang Layer1Engine.score (gate/engine.py), tren khong gian input day du cua model:
95 chieu (AE) hoac 90 (SVDD), vi chieu augmented (ae_recon_error, svdd_d_min...) la thu model dung
de quyet dinh. Doc tap R luc chay: retrieval/store.py (ReferenceStore).

CHAY (Protocol B):  python LAYER2/retrieval/reference_set.py                 # ae_xgboost va svdd_xgboost
                    python LAYER2/retrieval/reference_set.py --only ae_xgboost
                    python LAYER2/retrieval/reference_set.py --add-raw       # them x__/row__ vao R da co,
                                                                             # KHONG tinh lai SHAP; kiem mau
Build lai: tau_zd cu duoc giu o "novelty_previous_build" (khong dung); chay lai 02_calibrate_novelty.py.
--add-raw giu nguyen "novelty" vi vector SHAP khong doi.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):                  # chay truc tiep: python LAYER2/retrieval/reference_set.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

# Cap moi loai. Khong bao gio cham: moi loai tan cong co toi da 20K flow trong toan bo du lieu,
# nen R chua moi flow attack cua train_full.
CAP_PER_LABEL = 20_000
SEED = 42
BATCH = 20_000

# Benign KHONG nam trong reference set.
#
# Ly do (2026-09-27): pool nay duoc truy hoi bang chinh vector SHAP cua Layer 1,
# nen dung no de phu dinh Layer 1 la lap luan vong tron. Do thuc te tren 5 false
# positive: ca 5 deu khop ATTACK profile voi sim = 1.000 (4/5 khop Infiltration),
# khong cai nao khop Benign - vi nhung flow bi bao nham chinh la nhung flow co
# vector SHAP giong attack, trong khi label__Benign chi chua benign ma gate phan
# loai DUNG. Them vao do, nhan Benign cua CSE-CIC-IDS2018 khong dang tin: audit
# cua DistriNet ghi nhan attack that bi gan nhan Benign o dung hai ngay
# Infiltration.
#
# Vai tro cua RAG vi vay la MO TA loai tan cong cho alert, khong phai phan xu
# alert dung hay sai.
EXCLUDE_LABELS = ["Benign"]


def protocol_b_path(variant):
    return Path(config.REFERENCE_SETS_DIR) / "protocol_b" / f"{variant}.npz"


def protocol_a_path(fold, variant):
    return Path(config.REFERENCE_SETS_DIR) / "protocol_a" / fold / f"{variant}.npz"


def protocol_a_folds(variant):
    """[(fold, duong dan tap R)] cua cac fold da build cho variant."""
    root = Path(config.REFERENCE_SETS_DIR) / "protocol_a"
    return [(p.parent.name, p) for p in sorted(root.glob(f"*/{variant}.npz"))]


def save(npz_path, pools, feature_names, meta, raw=None, base_feature_names=None, keep_novelty=False):
    """Ghi tap R: pools = {Label_Text: (n, d)} -> khoa label__<nhan>; meta -> <file>.meta.json.

    raw = {Label_Text: (X_raw (n, 78), row_ids (n,))} cung thu tu dong voi pools -> khoa x__/row__.
    keep_novelty: giu "novelty" (tau_zd) cua meta cu (dung khi vector SHAP khong doi, vd --add-raw);
    neu False, tau_zd cu (neu co) duoc chuyen sang "novelty_previous_build" de khong bi mat.
    """
    out = Path(npz_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"label__{lab}": v for lab, v in pools.items()}
    for lab, (x, rows) in (raw or {}).items():
        assert len(x) == len(rows) == len(pools[lab]), f"{lab}: x/row/phi lech so dong"
        arrays[f"x__{lab}"] = np.asarray(x, dtype="float64")
        arrays[f"row__{lab}"] = np.asarray(rows, dtype="int64")
    if base_feature_names is not None:
        arrays["base_feature_names"] = np.array(list(base_feature_names), dtype=object)
    np.savez_compressed(out, feature_names=np.array(list(feature_names), dtype=object), **arrays)

    mpath = out.with_suffix(".meta.json")
    old = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    if old.get("novelty"):
        meta["novelty" if keep_novelty else "novelty_previous_build"] = old["novelty"]
    mpath.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def build_protocol_b(variant: str) -> Path:
    from gate.engine import Layer1Engine

    print(f"\n{'=' * 70}\n  TAP R (Protocol B) cho {variant}\n{'=' * 70}")
    l1 = Layer1Engine(variant=variant)

    df = pd.read_csv(config.TRAIN_FULL_CSV)
    feats = l1.base_feature_columns(df)
    print(f"  train_full.csv: {len(df):,} dong, {len(feats)} feature goc")

    rng = np.random.RandomState(SEED)
    pools, meta = {}, {}
    t_start = time.time()

    labels = [L for L in sorted(df["Label_Text"].unique())
              if L not in EXCLUDE_LABELS]
    print(f"  loai khoi R: {EXCLUDE_LABELS} -> con {len(labels)} nhan")

    raw = {}
    for label in labels:
        sub = df[df["Label_Text"] == label]
        n_have = len(sub)
        if n_have > CAP_PER_LABEL:
            idx = rng.choice(n_have, CAP_PER_LABEL, replace=False)
            sub = sub.iloc[idx]

        X = sub[feats].to_numpy(dtype="float64")
        raw[label] = (X, sub.index.to_numpy(dtype="int64"))     # gia tri tho + dong goc, cung thu tu voi SHAP

        # TreeExplainer theo lo de khong vo RAM
        parts = []
        for i in range(0, len(X), BATCH):
            _, sv = l1.score(X[i:i + BATCH])
            parts.append(np.asarray(sv, dtype="float32"))
        vecs = np.vstack(parts)

        pools[label] = vecs
        meta[label] = {"n_available": int(n_have), "n_kept": int(len(vecs))}
        print(f"    {label:<26}{n_have:>7,} -> {len(vecs):>6,}  shape={vecs.shape}")

    dt = time.time() - t_start
    total = sum(v.shape[0] for v in pools.values())
    out = save(protocol_b_path(variant), pools, l1.feature_names, {
        "variant": variant,
        "source": str(Path(config.TRAIN_FULL_CSV).name),
        "n_dims": int(l1.n_features),
        "n_base_features": int(l1.n_base),
        "cap_per_label": CAP_PER_LABEL,
        "excluded_labels": EXCLUDE_LABELS,
        "seed": SEED,
        "n_labels": len(pools),
        "n_vectors_total": int(total),
        "build_seconds": round(dt, 1),
        "labels": meta,
        "raw_values": {"keys": "x__<label> (78 base features, float64), row__<label> (row index in train_full.csv)",
                       "source": "written by build_protocol_b"},
    }, raw=raw, base_feature_names=feats)
    print(f"\n  {len(pools)} nhan | {total:,} vector | {l1.n_features} chieu | {dt:.0f}s")
    print(f"  -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def add_raw_protocol_b(variant: str, n_check: int = 300) -> Path:
    """Them x__/row__ vao tap R Protocol B da build, KHONG tinh lai SHAP.

    Dong cua moi loai trong R la cac dong cua loai do trong train_full.csv theo thu tu file (CAP khong
    bao gio cham, nen khong co lay mau). Ham dung lai chi so dong theo dung quy tac do, roi KIEM: tinh
    lai SHAP cua n_check dong ngau nhien bang Layer 1 hien tai va so voi vector da luu.
    """
    from gate.engine import Layer1Engine

    path = protocol_b_path(variant)
    data = np.load(path, allow_pickle=True)
    pools = {k[len("label__"):]: np.asarray(data[k]) for k in data.files if k.startswith("label__")}
    df = pd.read_csv(config.TRAIN_FULL_CSV)
    l1 = Layer1Engine(variant=variant)
    feats = l1.base_feature_columns(df)

    raw = {}
    for label, phi in pools.items():
        rows = np.flatnonzero(df["Label_Text"].to_numpy() == label)
        if len(rows) != len(phi):
            raise ValueError(f"{label}: R co {len(phi)} vector nhung train_full co {len(rows)} dong -> "
                             f"R khong duoc build bang quy tac hien tai, phai build lai (khong --add-raw)")
        raw[label] = (df.iloc[rows][feats].to_numpy(dtype="float64"), rows)

    # Kiem khop dong: SHAP tinh lai == SHAP da luu tren mau ngau nhien
    rng = np.random.RandomState(SEED)
    flat = [(lab, i) for lab, phi in pools.items() for i in range(len(phi))]
    pick = [flat[j] for j in rng.choice(len(flat), min(n_check, len(flat)), replace=False)]
    X = np.vstack([raw[lab][0][i] for lab, i in pick])
    _, sv = l1.score(X)
    stored = np.vstack([pools[lab][i] for lab, i in pick]).astype("float64")
    diff = float(np.abs(np.asarray(sv, dtype="float64") - stored).max())
    print(f"  kiem {len(pick)} dong: max |phi_tinh_lai - phi_da_luu| = {diff:.2e}")
    if diff > 1e-3:
        raise ValueError(f"vector SHAP khong khop (max diff {diff:.3g}): R va Layer 1 / thu tu dong lech nhau")

    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    meta["raw_values"] = {
        "keys": "x__<label> (78 base features, float64), row__<label> (row index in train_full.csv)",
        "source": "added by reference_set.py --add-raw without recomputing SHAP",
        "row_rule": "rows of each label in train_full.csv in file order (cap never binds)",
        "verified_rows": len(pick), "max_abs_phi_diff": diff,
    }
    out = save(path, pools, list(data["feature_names"]), meta, raw=raw, base_feature_names=feats,
               keep_novelty=True)
    print(f"  -> {out}  ({out.stat().st_size / 1e6:.1f} MB), giu novelty: {meta.get('novelty', {}).get('tau_zd')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(config.LAYER1_ARTIFACTS), default=None)
    ap.add_argument("--add-raw", action="store_true",
                    help="them gia tri tho + chi so dong vao R da co, khong tinh lai SHAP")
    args = ap.parse_args()

    variants = [args.only] if args.only else list(config.LAYER1_ARTIFACTS)
    if args.add_raw:
        made = [add_raw_protocol_b(v) for v in variants]
    else:
        made = [build_protocol_b(v) for v in variants]

    print(f"\n{'=' * 70}")
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    main()
