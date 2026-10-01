"""
Build 03_splits_v2/ tu 03_splits/ (v1) + Benign.parquet goc — KHONG doc lai raw
parquet cho Attack (v1 da train_test_split(stratify=Label_Text) dung cach roi).

Xem README.md cung thu muc de biet ly do va thiet ke day du.

Pipeline:
  1. Doc Benign.parquet goc (10,564,771 dong), sample BENIGN_CAP=700,000 dong moi
     (seed=42, KHAC lan sample cua v1 — day la 1 tap Benign moi hoan toan).
  2. Chia 700,000 Benign do thanh 2 pool CO DINH, KHONG CHONG LAP, 1 LAN DUY NHAT:
       - benign_train_pool (80% = 560,000 dong)
       - benign_val_pool   (20% = 140,000 dong)
  3. Voi moi label trong 14 attack_labels:
       a. Doc 03_splits/by_label/<label>/train_benign_attack.csv, LOC BO cac dong
          Label_Text == "Benign" (giu nguyen phan Attack, KHONG chia lai).
       b. Ghep attack_train_only + benign_train_pool -> train_benign_attack.csv (v2).
       c. Tuong tu cho val_binary.csv voi benign_val_pool.
       d. Copy nguyen ven test_zeroshot.csv (khong doi, khong lien quan Benign).
  4. Ghi train_benign_only.csv = benign_train_pool.
  5. Ghi split_manifest_v2.json (cap, ty le Benign%, checksum).

Assertion chan leak (bat buoc):
  - benign_train_pool va benign_val_pool khong co index nao trung nhau.
  - Sau khi loc, train_benign_attack.csv (v2) khong con dong Benign nao tu v1 sot lai
    (chi con benign_train_pool moi).
  - held_out_label khong xuat hien trong train_benign_attack.csv/val_binary.csv (v2).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
V1_SPLITS_DIR = BASE_DIR / "03_splits"
V1_BY_LABEL_DIR = V1_SPLITS_DIR / "by_label"
BENIGN_PARQUET_PATH = BASE_DIR / "02_cleaning" / "clean_data" / "by_label" / "Benign.parquet"
LABELS_CONFIG_PATH = BASE_DIR / "config" / "labels.json"

V2_SPLITS_DIR = Path(__file__).resolve().parent
V2_BY_LABEL_DIR = V2_SPLITS_DIR / "by_label"
MANIFEST_PATH = V2_SPLITS_DIR / "split_manifest_v2.json"
TRAIN_BENIGN_ONLY_PATH = V2_SPLITS_DIR / "train_benign_only.csv"

BENIGN_CAP = 700_000
BENIGN_TRAIN_FRACTION = 0.80
RANDOM_SEED = 42


def load_label_config() -> tuple[list[str], str]:
    with open(LABELS_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["attack_labels"], cfg["benign_label"]


def file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_new_benign_pools(benign_label: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Doc Benign.parquet goc, sample BENIGN_CAP, chia 80/20 CO DINH mot lan duy nhat."""
    assert BENIGN_PARQUET_PATH.exists(), f"Khong tim thay {BENIGN_PARQUET_PATH}"

    df = pd.read_parquet(BENIGN_PARQUET_PATH)
    n_before = len(df)
    print(f"Benign.parquet goc: {n_before:,} dong")

    df_capped = df.sample(n=min(BENIGN_CAP, n_before), random_state=RANDOM_SEED).reset_index(drop=True)
    df_capped["Label"] = 0
    df_capped["Label_Text"] = benign_label
    print(f"Sau cap BENIGN_CAP={BENIGN_CAP:,}: {len(df_capped):,} dong")

    rng = np.random.RandomState(RANDOM_SEED)
    shuffled_idx = rng.permutation(len(df_capped))
    n_train = int(len(df_capped) * BENIGN_TRAIN_FRACTION)

    train_idx = shuffled_idx[:n_train]
    val_idx = shuffled_idx[n_train:]

    assert len(set(train_idx) & set(val_idx)) == 0, "LEAK: benign_train_pool va benign_val_pool trung index"

    benign_train_pool = df_capped.iloc[train_idx].reset_index(drop=True)
    benign_val_pool = df_capped.iloc[val_idx].reset_index(drop=True)

    print(f"benign_train_pool = {len(benign_train_pool):,} dong (80%)")
    print(f"benign_val_pool   = {len(benign_val_pool):,} dong (20%)")

    return benign_train_pool, benign_val_pool, n_before


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "Label" in df.columns and "Label_Text" in df.columns, f"{path}: thieu cot Label/Label_Text"
    return df


def build_split_v2_for_label(
    held_out_label: str,
    benign_label: str,
    benign_train_pool: pd.DataFrame,
    benign_val_pool: pd.DataFrame,
) -> dict:
    v1_label_dir = V1_BY_LABEL_DIR / held_out_label
    v2_label_dir = V2_BY_LABEL_DIR / held_out_label
    v2_label_dir.mkdir(parents=True, exist_ok=True)

    # --- train_benign_attack.csv: loc bo Benign cu, ghep Benign moi ---
    v1_train_df = load_csv(v1_label_dir / "train_benign_attack.csv")
    attack_train_only = v1_train_df[v1_train_df["Label_Text"] != benign_label].reset_index(drop=True)
    assert (attack_train_only["Label_Text"] == benign_label).sum() == 0, "Loc Benign cu that bai"

    train_df_v2 = pd.concat([benign_train_pool, attack_train_only], axis=0, ignore_index=True)
    assert held_out_label not in set(train_df_v2["Label_Text"].unique()), (
        f"LEAK: {held_out_label} xuat hien trong train_benign_attack.csv (v2)"
    )
    train_df_v2.to_csv(v2_label_dir / "train_benign_attack.csv", index=False)

    # --- val_binary.csv: tuong tu ---
    v1_val_df = load_csv(v1_label_dir / "val_binary.csv")
    attack_val_only = v1_val_df[v1_val_df["Label_Text"] != benign_label].reset_index(drop=True)
    assert (attack_val_only["Label_Text"] == benign_label).sum() == 0, "Loc Benign cu that bai"

    val_df_v2 = pd.concat([benign_val_pool, attack_val_only], axis=0, ignore_index=True)
    assert held_out_label not in set(val_df_v2["Label_Text"].unique()), (
        f"LEAK: {held_out_label} xuat hien trong val_binary.csv (v2)"
    )
    val_df_v2.to_csv(v2_label_dir / "val_binary.csv", index=False)

    # --- test_zeroshot.csv: copy nguyen ven, khong doi ---
    v1_test_path = v1_label_dir / "test_zeroshot.csv"
    v2_test_path = v2_label_dir / "test_zeroshot.csv"
    test_df = load_csv(v1_test_path)
    test_labels_text = set(test_df["Label_Text"].unique())
    assert test_labels_text == {held_out_label}, (
        f"test_zeroshot.csv cho {held_out_label} chua nhan la: {test_labels_text}"
    )
    test_df.to_csv(v2_test_path, index=False)

    n_benign_train = int((train_df_v2["Label_Text"] == benign_label).sum())
    n_attack_train = len(train_df_v2) - n_benign_train
    benign_pct = n_benign_train / len(train_df_v2) * 100

    return {
        "held_out_label": held_out_label,
        "n_train_benign_attack": len(train_df_v2),
        "n_benign_train": n_benign_train,
        "n_attack_train": n_attack_train,
        "benign_pct_train": round(benign_pct, 2),
        "n_val_binary": len(val_df_v2),
        "n_test_zeroshot": len(test_df),
        "checksums": {
            "train_benign_attack.csv": file_checksum(v2_label_dir / "train_benign_attack.csv"),
            "val_binary.csv": file_checksum(v2_label_dir / "val_binary.csv"),
            "test_zeroshot.csv": file_checksum(v2_test_path),
        },
    }


def main() -> None:
    attack_labels, benign_label = load_label_config()
    print(f"{len(attack_labels)} attack label (leave-one-out, khong doi so voi v1): {attack_labels}")
    print(f"BENIGN_CAP moi = {BENIGN_CAP:,} (v1 la 100,000)")

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "source_v1": str(V1_SPLITS_DIR),
        "benign_cap": BENIGN_CAP,
        "benign_train_fraction": BENIGN_TRAIN_FRACTION,
        "random_seed": RANDOM_SEED,
        "note": (
            "v2: BENIGN_CAP 100K->700K de dat ty le Benign~83% (CIC-IDS-2018 goc); "
            "benign_train_pool/benign_val_pool tach CO DINH 1 lan, khong chong lap "
            "(fix leak cua v1, xem README.md)."
        ),
        "splits": {},
    }

    benign_train_pool, benign_val_pool, n_benign_before = build_new_benign_pools(benign_label)
    manifest["benign_before_cap"] = n_benign_before
    manifest["benign_train_pool_size"] = len(benign_train_pool)
    manifest["benign_val_pool_size"] = len(benign_val_pool)

    V2_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    benign_train_pool.to_csv(TRAIN_BENIGN_ONLY_PATH, index=False)
    print(f"\nDa ghi {TRAIN_BENIGN_ONLY_PATH} ({len(benign_train_pool):,} dong)")

    for held_out_label in attack_labels:
        print(f"\n=== Dung split v2: held-out = {held_out_label} ===")
        split_info = build_split_v2_for_label(held_out_label, benign_label, benign_train_pool, benign_val_pool)
        manifest["splits"][held_out_label] = split_info
        print(
            f"  train_benign_attack={split_info['n_train_benign_attack']:,} "
            f"(benign={split_info['n_benign_train']:,} = {split_info['benign_pct_train']}%, "
            f"attack={split_info['n_attack_train']:,})  "
            f"val_binary={split_info['n_val_binary']:,}  "
            f"test_zeroshot={split_info['n_test_zeroshot']:,}"
        )

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nDa ghi split manifest v2: {MANIFEST_PATH}")
    print(f"Hoan tat: {len(attack_labels)} split v2 da duoc tao trong {V2_BY_LABEL_DIR}")


if __name__ == "__main__":
    main()
