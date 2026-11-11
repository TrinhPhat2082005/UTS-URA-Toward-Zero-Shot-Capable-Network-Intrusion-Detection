"""
Tap tham chieu R (paper Section 3.5): vector SHAP cua moi flow attack cua tap train, gom theo loai.

    <UTS_DATA>/artifacts/reference_sets/protocol_b/<variant>.npz          Protocol B (train_full.csv), file nay build
    <UTS_DATA>/artifacts/reference_sets/protocol_a/<fold>/<variant>.npz   Protocol A, tung fold LOAO
                                                                          (experiments/rq3_retrieval/01_build_loao_pools.py)

Moi file: khoa label__<Label_Text> -> (n, d) float32 va feature_names; kem <file>.meta.json (nguon,
so vector moi loai, ... va "novelty" -> "tau_zd" sau khi chay 02_calibrate_novelty.py).
Vector SHAP tinh bang Layer1Engine.score (gate/engine.py), tren khong gian input day du cua model:
95 chieu (AE) hoac 90 (SVDD), vi chieu augmented (ae_recon_error, svdd_d_min...) la thu model dung
de quyet dinh.

CHAY (Protocol B):  python LAYER2/retrieval/reference_set.py                 # ae_xgboost va svdd_xgboost
                    python LAYER2/retrieval/reference_set.py --only ae_xgboost
Build lai se ghi de .meta.json: chay lai 02_calibrate_novelty.py de co tau_zd.
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


def save(npz_path, pools, feature_names, meta):
    """Ghi tap R: pools = {Label_Text: (n, d)} -> khoa label__<nhan>; meta -> <file>.meta.json."""
    out = Path(npz_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, feature_names=np.array(list(feature_names), dtype=object),
                        **{f"label__{lab}": v for lab, v in pools.items()})
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
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

    for label in labels:
        sub = df[df["Label_Text"] == label]
        n_have = len(sub)
        if n_have > CAP_PER_LABEL:
            idx = rng.choice(n_have, CAP_PER_LABEL, replace=False)
            sub = sub.iloc[idx]

        X = sub[feats].to_numpy(dtype="float64")

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
    })
    print(f"\n  {len(pools)} nhan | {total:,} vector | {l1.n_features} chieu | {dt:.0f}s")
    print(f"  -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(config.LAYER1_ARTIFACTS), default=None)
    args = ap.parse_args()

    variants = [args.only] if args.only else list(config.LAYER1_ARTIFACTS)
    made = [build_protocol_b(v) for v in variants]

    print(f"\n{'=' * 70}")
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    main()
