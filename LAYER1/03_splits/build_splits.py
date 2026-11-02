"""
Build split leave-one-attack-out (Protocol A) — MOT script duy nhat, thay cho 2 buoc cu
03_splits/build_leave_one_out_splits.py (v1) + 03_splits_v2/build_v2_from_v1.py (v2).
Cho ra dung tung byte bo split dang dung de train checkpoint (zeroshot_benchmark/03_splits_v2/):
cuoi script so sha256 voi split_manifest_reference.json (manifest goc, sinh tren Kaggle).

Input : <UTS_BENCH>/02_cleaning/clean_data/by_label/<label>.parquet   (output cua 02_cleaning)
        config/labels.json
Output: <UTS_BENCH>/03_splits_v2/
          train_benign_only.csv            benign train pool (560,000 dong) — train AE/SVDD/CLAD
          by_label/<held_out>/
            train_benign_attack.csv        benign train pool + 80% cua 13 attack con lai
            val_binary.csv                 benign val pool (140,000) + 20% cua 13 attack con lai
            test_zeroshot.csv              toan bo attack bi giu lai (da cap), khong benign
          split_manifest.json              cap, so dong, sha256 tung file

Cac buoc (dung thu tu da tao split hien tai):
  1. Moi attack class lay toi da ATTACK_CAP_PER_CLASS dong (sample seed 42); class it hon giu nguyen.
  2. Chia attack 80/20 cho tung fold: train_test_split(stratify=Label_Text, seed 42) tren
     [BENIGN_CAP_V1 benign + 13 attack con lai], y het v1. 100K benign nay CHI de tai tao
     dung flow attack nao roi vao train/val (train_test_split xao tron ca tap gop); bi bo ngay
     sau khi chia, khong ghi ra dau. Bo no di se doi cach chia attack so voi checkpoint da train.
  3. Benign: sample BENIGN_CAP dong moi (seed 42), chia CO DINH 80/20 thanh benign_train_pool /
     benign_val_pool, dung chung cho moi fold.
  4. Moi fold: train = train pool + attack train; val = val pool + attack val; test = attack bi giu lai.
     Phan attack (va test) di qua mot vong ghi/doc CSV truoc khi ghi: v2 da doc lai file CSV cua v1
     bang pd.read_csv, ma bo parse float mac dinh khong giu dung chu so cuoi (0.07355545738417563 ->
     0.0735554573841756). Split hien tai mang dau vet do; bo vong nay thi file lech byte (gia tri
     lech ~1e-16), benign pool thi khong qua vong nay.

Vi sao benign tach 2 pool (ly do cua ban v2):
  - v1 dung BENIGN_CAP=100K -> benign chi ~35-38% moi fold, nguoc CIC-IDS-2018 goc (~83%).
  - v1 lay benign cua val_binary tu cung pool voi train_benign_only.csv -> AE/SVDD/CLAD da "thay"
    chinh benign dung de chon nguong va danh gia (leak). 700K + 2 pool khong chong lan sua ca hai.

CHAY:  python build_splits.py [--clean-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent                 # LAYER1/ (hoac zeroshot_benchmark/)
BENCH = Path(os.environ.get("UTS_BENCH", BASE_DIR))
CLEAN_BY_LABEL_DIR = BENCH / "02_cleaning" / "clean_data" / "by_label"
OUT_DIR = BENCH / "03_splits_v2"
LABELS_CONFIG_PATH = BASE_DIR / "config" / "labels.json"
REFERENCE_MANIFEST_PATH = Path(__file__).resolve().parent / "split_manifest_reference.json"

RANDOM_SEED = 42
VAL_FRACTION = 0.20
ATTACK_CAP_PER_CLASS = 20_000
BENIGN_CAP_V1 = 100_000          # chi de tai tao cach chia attack (buoc 2), khong ghi ra
BENIGN_CAP = 700_000
BENIGN_TRAIN_FRACTION = 0.80


def load_label_config() -> tuple[list[str], str]:
    with open(LABELS_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["attack_labels"], cfg["benign_label"]


def read_parquet_sample(path: Path, n: int, seed: int) -> pd.DataFrame:
    """= pd.read_parquet(path).sample(n=n, random_state=seed).reset_index(drop=True), nhung chi
    doc dung n dong can lay theo tung row group (Benign.parquet ~10.5M dong khong vua RAM 16 GB).
    DataFrame.sample chon vi tri bang RandomState(seed).choice(so_dong, n, replace=False)."""
    pf = pq.ParquetFile(path)
    locs = np.random.RandomState(seed).choice(pf.metadata.num_rows, size=n, replace=False)
    order = np.argsort(locs, kind="stable")
    sorted_locs = locs[order]
    parts, start = [], 0
    for g in range(pf.num_row_groups):
        size = pf.metadata.row_group(g).num_rows
        lo, hi = np.searchsorted(sorted_locs, [start, start + size])
        if hi > lo:
            parts.append(pf.read_row_group(g).take(pa.array(sorted_locs[lo:hi] - start)))
        start += size
    df_sorted = pa.concat_tables(parts).to_pandas()
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))                 # dong thu j cua ket qua = locs[j]
    return df_sorted.iloc[inv].reset_index(drop=True)


def load_capped(path: Path, cap: int, always_sample: bool) -> tuple[pd.DataFrame, int]:
    """Cap mot nhan. v1 (attack, benign 100K) chi sample khi vuot cap; v2 (benign 700K) luon
    sample min(cap, n) — giu dung ca hai hanh vi de ket qua khop byte."""
    assert path.exists(), f"Khong tim thay {path} — chay 02_cleaning/clean_and_relabel.ipynb truoc."
    n_before = pq.ParquetFile(path).metadata.num_rows
    if always_sample or n_before > cap:
        df = read_parquet_sample(path, min(cap, n_before), RANDOM_SEED)
    else:
        df = pd.read_parquet(path)
    return df, n_before


def with_labels(df: pd.DataFrame, label_text: str, is_benign: bool) -> pd.DataFrame:
    df["Label"] = 0 if is_benign else 1
    df["Label_Text"] = label_text
    return df


def csv_roundtrip(df: pd.DataFrame) -> pd.DataFrame:
    """Ghi CSV roi doc lai, nhu v2 doc lai file cua v1 (xem buoc 4 o docstring)."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf)


def file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_benign_pools(benign_path: Path, benign_label: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Sample BENIGN_CAP benign, chia 80/20 CO DINH mot lan duy nhat cho moi fold."""
    df, n_before = load_capped(benign_path, BENIGN_CAP, always_sample=True)
    df = with_labels(df, benign_label, is_benign=True)
    shuffled_idx = np.random.RandomState(RANDOM_SEED).permutation(len(df))
    n_train = int(len(df) * BENIGN_TRAIN_FRACTION)
    train_idx, val_idx = shuffled_idx[:n_train], shuffled_idx[n_train:]
    assert len(set(train_idx) & set(val_idx)) == 0, "LEAK: benign_train_pool va benign_val_pool trung dong"
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True), n_before


def split_attack_for_fold(held_out: str, benign_v1: pd.DataFrame, attacks: dict[str, pd.DataFrame],
                          benign_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phan attack train/val cua fold, chia y het v1 (xem buoc 2 o docstring)."""
    combined = pd.concat([benign_v1] + [df for lbl, df in attacks.items() if lbl != held_out],
                         axis=0, ignore_index=True)
    train_df, val_df = train_test_split(combined, test_size=VAL_FRACTION,
                                        stratify=combined["Label_Text"], random_state=RANDOM_SEED)
    train_df, val_df = csv_roundtrip(train_df), csv_roundtrip(val_df)   # ca file v1, ke ca benign 100K
    attack_train = train_df[train_df["Label_Text"] != benign_label].reset_index(drop=True)
    attack_val = val_df[val_df["Label_Text"] != benign_label].reset_index(drop=True)
    return attack_train, attack_val


def verify_against_reference(manifest: dict) -> None:
    """So sha256 voi manifest goc — khop het nghia la split trung byte voi split da dung."""
    if not REFERENCE_MANIFEST_PATH.exists():
        print("Khong co split_manifest_reference.json — bo qua buoc so checksum.")
        return
    ref = json.loads(REFERENCE_MANIFEST_PATH.read_text(encoding="utf-8"))["splits"]
    diff = [f"{fold}/{name}" for fold, info in manifest["splits"].items()
            for name, sha in info["checksums"].items() if ref.get(fold, {}).get("checksums", {}).get(name) != sha]
    n = sum(len(info["checksums"]) for info in manifest["splits"].values())
    print(f"\nSo voi manifest goc: {n - len(diff)}/{n} file khop" + (f"; KHAC: {diff}" if diff else ""))


def main() -> None:
    global CLEAN_BY_LABEL_DIR, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", type=Path, default=CLEAN_BY_LABEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    CLEAN_BY_LABEL_DIR, OUT_DIR = args.clean_dir, args.out_dir

    attack_labels, benign_label = load_label_config()
    print(f"{len(attack_labels)} held-out label; attack cap {ATTACK_CAP_PER_CLASS:,}, benign cap {BENIGN_CAP:,}")
    manifest = {"generated_at": datetime.now().isoformat(), "random_seed": RANDOM_SEED,
                "val_fraction": VAL_FRACTION, "attack_cap_per_class": ATTACK_CAP_PER_CLASS,
                "benign_cap_v1_for_attack_split": BENIGN_CAP_V1, "benign_cap": BENIGN_CAP,
                "benign_train_fraction": BENIGN_TRAIN_FRACTION, "source": str(CLEAN_BY_LABEL_DIR),
                "raw_row_counts": {}, "splits": {}}

    attacks: dict[str, pd.DataFrame] = {}
    for lbl in attack_labels:
        df, n_before = load_capped(CLEAN_BY_LABEL_DIR / f"{lbl}.parquet", ATTACK_CAP_PER_CLASS, always_sample=False)
        attacks[lbl] = with_labels(df, lbl, is_benign=False)
        manifest["raw_row_counts"][lbl] = {"before_cap": n_before, "after_cap": len(df)}
        print(f"  {lbl:<26} {n_before:>10,} -> {len(df):>7,}")

    benign_path = CLEAN_BY_LABEL_DIR / f"{benign_label}.parquet"
    benign_v1, _ = load_capped(benign_path, BENIGN_CAP_V1, always_sample=False)
    benign_v1 = with_labels(benign_v1, benign_label, is_benign=True)
    benign_train_pool, benign_val_pool, n_benign = build_benign_pools(benign_path, benign_label)
    manifest["raw_row_counts"][benign_label] = {"before_cap": n_benign, "after_cap": BENIGN_CAP}
    manifest["benign_train_pool_size"], manifest["benign_val_pool_size"] = len(benign_train_pool), len(benign_val_pool)
    print(f"  {benign_label:<26} {n_benign:>10,} -> train pool {len(benign_train_pool):,} / val pool {len(benign_val_pool):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    benign_train_pool.to_csv(OUT_DIR / "train_benign_only.csv", index=False)

    for held_out in attack_labels:
        fold_dir = OUT_DIR / "by_label" / held_out
        fold_dir.mkdir(parents=True, exist_ok=True)
        attack_train, attack_val = split_attack_for_fold(held_out, benign_v1, attacks, benign_label)
        train_df = pd.concat([benign_train_pool, attack_train], axis=0, ignore_index=True)
        val_df = pd.concat([benign_val_pool, attack_val], axis=0, ignore_index=True)
        test_df = csv_roundtrip(attacks[held_out])
        assert held_out not in set(train_df["Label_Text"]), f"LEAK: {held_out} trong train_benign_attack.csv"
        assert held_out not in set(val_df["Label_Text"]), f"LEAK: {held_out} trong val_binary.csv"
        assert set(test_df["Label_Text"]) == {held_out}, f"test_zeroshot.csv cua {held_out} chua nhan la"

        files = {"train_benign_attack.csv": train_df, "val_binary.csv": val_df, "test_zeroshot.csv": test_df}
        for name, df in files.items():
            df.to_csv(fold_dir / name, index=False)
        n_benign_train = int((train_df["Label_Text"] == benign_label).sum())
        manifest["splits"][held_out] = {
            "n_train_benign_attack": len(train_df), "n_benign_train": n_benign_train,
            "n_attack_train": len(train_df) - n_benign_train,
            "benign_pct_train": round(n_benign_train / len(train_df) * 100, 2),
            "n_val_binary": len(val_df), "n_test_zeroshot": len(test_df),
            "checksums": {name: file_checksum(fold_dir / name) for name in files}}
        print(f"  {held_out:<26} train {len(train_df):,}  val {len(val_df):,}  test {len(test_df):,}")

    with open(OUT_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nDa ghi {len(attack_labels)} fold vao {OUT_DIR}")
    verify_against_reference(manifest)


if __name__ == "__main__":
    main()
