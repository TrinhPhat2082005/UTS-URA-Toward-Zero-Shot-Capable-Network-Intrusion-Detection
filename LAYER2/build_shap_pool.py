"""
Build pool vector SHAP cho RAG k-NN — chay cho ca 2 variant.

CHAY:  python build_shap_pool.py              # ca ae_xgboost lan svdd_xgboost
       python build_shap_pool.py --only ae_xgboost

CHUOI
-----
    train_full.csv -> Layer 1 (scale, clip, augment) -> TreeExplainer
        -> gom vector SHAP theo Label_Text -> luu .npz

Pool nay la "reference set R" trong Methodology 3.5: moi nhan mot mang vector
SHAP, va luc truy hoi thi so cosine giua vector cua flow dang xet voi TOAN BO
pool cua tung nhan (lay max), khong phai voi mot centroid trung binh.

HAI DIEM KHAC POOL CU
---------------------
1. So chieu: 95 (AE) hoac 90 (SVDD) thay vi 78. Vector SHAP bay gio bao gom ca
   chieu tang cuong (ae_recon_error, svdd_d_min...) vi do chinh la thu model
   dung de quyet dinh. Do trong mot mau thu, ae_recon_error dung thu 2 trong
   top dong gop - khong phai feature thua.

2. KHONG co label__Benign - xem EXCLUDE_LABELS ben duoi. Pool chi chua 14 loai
   tan cong, nen cau hoi ma RAG tra loi la "alert nay giong loai tan cong nao",
   khong phai "alert nay dung hay sai".

CAP MAU
-------
20,000 moi nhan (Benign 560K se bi cap xuong). Giu nguyen ty le cho cac nhan
nho hon cap. Cap de pool khong phinh va de cosine khong bi Benign ap dao.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config      # noqa: E402
import pipeline    # noqa: E402

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


def build_one(variant: str) -> Path:
    print(f"\n{'=' * 70}\n  POOL cho {variant}\n{'=' * 70}")

    l1 = pipeline.Layer1Engine(variant=variant)

    df = pd.read_csv(config.TRAIN_FULL_CSV)
    feats = l1.base_feature_columns(df)
    print(f"  train_full.csv: {len(df):,} dong, {len(feats)} feature goc")

    rng = np.random.RandomState(SEED)
    pools, meta = {}, {}
    t_start = time.time()

    labels = [L for L in sorted(df["Label_Text"].unique())
              if L not in EXCLUDE_LABELS]
    print(f"  loai khoi pool: {EXCLUDE_LABELS} -> con {len(labels)} nhan")

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

        key = f"label__{label}"
        pools[key] = vecs
        meta[label] = {"n_available": int(n_have), "n_kept": int(len(vecs))}
        print(f"    {label:<26}{n_have:>7,} -> {len(vecs):>6,}  shape={vecs.shape}")

    dt = time.time() - t_start

    out = Path(config.ARTIFACTS_DIR) / f"shap_pool_{variant}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        feature_names=np.array(l1.feature_names, dtype=object),
        **pools,
    )

    total = sum(v.shape[0] for v in pools.values())
    print(f"\n  {len(pools)} nhan | {total:,} vector | {l1.n_features} chieu "
          f"| {dt:.0f}s")
    print(f"  -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    with open(out.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump({
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
        }, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(config.LAYER1_ARTIFACTS), default=None)
    args = ap.parse_args()

    variants = [args.only] if args.only else list(config.LAYER1_ARTIFACTS)
    made = [build_one(v) for v in variants]

    print(f"\n{'=' * 70}")
    for p in made:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
