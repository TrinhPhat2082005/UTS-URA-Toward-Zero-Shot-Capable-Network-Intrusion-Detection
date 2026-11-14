"""
Chuan bi Kaggle dataset cho notebook vlm_run.ipynb (chay local MOT lan).

    python LAYER2/experiments/rq4_vlm/prepare_kaggle_dataset.py [--per-class 10] [--seed 42]

1. Chay Layer 1 (Protocol B, config.VARIANT) tren val_full.csv, giu cac dong TAN CONG ma Layer 1 bao dong (TP).
2. Lay ngau nhien --per-class dong moi nhan (seed co dinh; nhan it hon thi lay het) -> samples.csv: dong THO goc
   (78 feature + metadata) + sample_id = <Label_Text>__r<chi so dong trong val_full.csv> + row_idx.
   Notebook chay lai Layer 1 / detect tu dong tho nay, nen ket qua chon mau duoc kiem lai tren Kaggle.
3. Chep checkpoint Layer 1 va tap R (+ meta co tau_zd) theo dung bo cuc ma config.py doc qua UTS_DATA:

    <UTS_DATA>/kaggle_upload/layer2-vlm-run/
        samples.csv
        artifacts/layer1/<variant>/{model.pkl, scaler.pkl, ae_model.pt, feature_names.json, metrics.json}
        artifacts/reference_sets/protocol_b/<variant>.npz, <variant>.meta.json
        README.txt

Upload thu muc nay thanh MOT Kaggle dataset (vd "layer2-vlm-run"); trong notebook dat UTS_DATA tro vao do.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from gate.engine import Layer1Engine  # noqa: E402


def tp_alerts(l1, csv_path, chunk=20000):
    df = pd.read_csv(csv_path)
    feats = l1.base_feature_columns(df)
    proba = np.concatenate([l1.predict_proba(df[feats].iloc[i:i + chunk].to_numpy("float64"))
                            for i in range(0, len(df), chunk)])
    df["row_idx"] = np.arange(len(df))
    df["_p"] = proba
    alerts = df[(df["_p"] >= l1.threshold) & (df["Label"] == 1)]
    print(f"{Path(csv_path).name}: {len(df):,} dong -> {len(alerts):,} alert TP (nguong {l1.threshold:.4f})")
    return alerts


def stratified(alerts, per_class, seed):
    parts = []
    for lab, g in alerts.groupby("Label_Text"):
        k = min(per_class, len(g))
        parts.append(g.sample(k, random_state=seed))
        if k < per_class:
            print(f"  {lab:<26} chi co {len(g)} alert TP")
    out = pd.concat(parts).sort_values(["Label_Text", "row_idx"])
    out.insert(0, "sample_id", [f"{l}__r{r}" for l, r in zip(out["Label_Text"], out["row_idx"])])
    return out.drop(columns=["_p"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=config.PER_CLASS)
    ap.add_argument("--seed", type=int, default=config.SAMPLE_SEED)
    ap.add_argument("--out", default=os.path.join(config.DATA_ROOT, "kaggle_upload", "layer2-vlm-run"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    l1 = Layer1Engine(config.VARIANT)
    samples = stratified(tp_alerts(l1, config.VAL_BINARY_CSV), args.per_class, args.seed)
    samples.to_csv(out / "samples.csv", index=False)
    print(f"-> {out / 'samples.csv'}: {len(samples)} mau, {samples['Label_Text'].nunique()} nhan")

    art = config.LAYER1_ARTIFACTS[config.VARIANT]
    l1_dir = out / "artifacts" / "layer1" / config.VARIANT
    l1_dir.mkdir(parents=True, exist_ok=True)
    for key in ("model", "scaler", "encoder", "features", "metrics"):
        shutil.copy2(art[key], l1_dir / Path(art[key]).name)
    r_dir = out / "artifacts" / "reference_sets" / "protocol_b"
    r_dir.mkdir(parents=True, exist_ok=True)
    npz = Path(config.REFERENCE_SET_NPZ)
    for p in (npz, npz.with_suffix(".meta.json")):
        shutil.copy2(p, r_dir / p.name)

    meta = json.loads(npz.with_suffix(".meta.json").read_text(encoding="utf-8"))
    (out / "README.txt").write_text(
        f"Layer 2 VLM run inputs (variant {config.VARIANT}).\n"
        f"samples.csv: {len(samples)} raw val_full rows, {args.per_class} true-positive alerts per label, "
        f"seed {args.seed}; sample_id = <Label_Text>__r<row index in val_full.csv>.\n"
        f"artifacts/layer1/{config.VARIANT}: Layer 1 checkpoint (threshold {l1.threshold:.4f}).\n"
        f"artifacts/reference_sets/protocol_b: reference set R ({meta['n_vectors_total']} vectors), "
        f"tau_zd {meta['novelty']['tau_zd']:.4f}.\n", encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"-> {out} ({size / 1e6:.1f} MB): upload thanh mot Kaggle dataset")


if __name__ == "__main__":
    main()
