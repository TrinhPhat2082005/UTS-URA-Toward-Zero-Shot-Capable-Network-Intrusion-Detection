"""
Sinh split leave-one-attack-out tu clean_data/by_label/*.parquet (output cua
02_cleaning/clean_and_relabel.ipynb — data AWS S3 goc, da lam sach + chuan hoa nhan,
audit day du qua cleaning_log.json). KHONG cleaning lai o day, chi doc + cap sample + split.

Quy uoc cot nhan:
  - Label      : binary, 0 = benign, 1 = anomaly/attack -> dung de train.
  - Label_Text : ten nhan chuan hoa (Benign, Bot, DDoS_attacks_LOIC_HTTP, ...) -> CHI de
                 audit/RAG, PHAI drop truoc khi dua vao model train.

Sampling cap (theo khao sat CHCG — MDPI Sensors 2024, PMC11014011 — ap dung cho CIC-IDS-2018):
  - Benign: cap toi da BENIGN_CAP dong (lay ngau nhien co seed co dinh).
  - Moi attack class: cap toi da ATTACK_CAP_PER_CLASS dong; class nao it hon thi lay het
    (vd SQL Injection, Brute_Force_XSS chi vai tram dong — xem cleaning_log.json).

Output (khong ghi lai train_benign_only.csv cho tung label — CHI 1 ban dung chung cho
ca 14 lan eval AE-Only/SVDD-Only, dat o 03_splits/train_benign_only.csv):
  03_splits/
    train_benign_only.csv        # toan bo Benign (da cap) — train AE/SVDD MOT LAN DUY NHAT
    by_label/<label>/
      train_benign_attack.csv    # 80% cua (Benign + 13 attack con lai), stratified theo Label
      val_binary.csv             # 20% con lai
      test_zeroshot.csv          # toan bo dong cua attack bi giu lai (KHONG kem benign)

Assertion chan leak (bat buoc, khong duoc bo qua neu that bai):
  - held_out_label khong xuat hien trong train_benign_attack.csv / val_binary.csv
  - test_zeroshot.csv CHI chua held_out_label, khong benign, khong attack khac
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_BY_LABEL_DIR = BASE_DIR / "02_cleaning" / "clean_data" / "by_label"
LABELS_CONFIG_PATH = BASE_DIR / "config" / "labels.json"
SPLITS_OUT_DIR = Path(__file__).resolve().parent
BY_LABEL_DIR = SPLITS_OUT_DIR / "by_label"
MANIFEST_PATH = SPLITS_OUT_DIR / "split_manifest.json"
TRAIN_BENIGN_ONLY_PATH = SPLITS_OUT_DIR / "train_benign_only.csv"

VAL_FRACTION = 0.20
RANDOM_SEED = 42
BENIGN_CAP = 100_000
ATTACK_CAP_PER_CLASS = 20_000


def load_label_config() -> tuple[dict[str, str], list[str], str]:
    with open(LABELS_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["label_map"], cfg["attack_labels"], cfg["benign_label"]


def load_and_cap_label_parquet(standard_label: str, is_benign: bool, cap: int) -> tuple[pd.DataFrame, int, int]:
    """Doc 1 file clean_data/by_label/<standard_label>.parquet (da sach san), gan
    Label/Label_Text, cap sample. Tra ve (df, n_before_cap, n_after_cap)."""
    path = CLEAN_BY_LABEL_DIR / f"{standard_label}.parquet"
    assert path.exists(), (
        f"Khong tim thay {path} — chay 02_cleaning/clean_and_relabel.ipynb truoc "
        f"(can output by_label/{standard_label}.parquet)."
    )

    df = pd.read_parquet(path)
    n_before = len(df)

    if n_before > cap:
        df = df.sample(n=cap, random_state=RANDOM_SEED).reset_index(drop=True)
    n_after = len(df)

    df["Label"] = 0 if is_benign else 1
    df["Label_Text"] = standard_label

    return df, n_before, n_after


def file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_split_for_label(
    held_out_label: str,
    benign_df: pd.DataFrame,
    other_attack_dfs: dict[str, pd.DataFrame],
    heldout_df: pd.DataFrame,
) -> dict:
    label_dir = BY_LABEL_DIR / held_out_label
    label_dir.mkdir(parents=True, exist_ok=True)

    train_benign_attack_path = label_dir / "train_benign_attack.csv"
    val_binary_path = label_dir / "val_binary.csv"
    test_zeroshot_path = label_dir / "test_zeroshot.csv"

    combined_df = pd.concat([benign_df] + list(other_attack_dfs.values()), axis=0, ignore_index=True)
    assert held_out_label not in set(combined_df["Label_Text"].unique()), (
        f"LEAK: {held_out_label} xuat hien trong tap combined truoc khi split train/val — bug trong code goi ham."
    )

    train_df, val_df = train_test_split(
        combined_df,
        test_size=VAL_FRACTION,
        stratify=combined_df["Label_Text"],
        random_state=RANDOM_SEED,
    )

    train_labels_text = set(train_df["Label_Text"].unique())
    val_labels_text = set(val_df["Label_Text"].unique())
    assert held_out_label not in train_labels_text, f"LEAK: {held_out_label} xuat hien trong train_benign_attack.csv"
    assert held_out_label not in val_labels_text, f"LEAK: {held_out_label} xuat hien trong val_binary.csv"

    train_df.to_csv(train_benign_attack_path, index=False)
    val_df.to_csv(val_binary_path, index=False)

    test_labels_text = set(heldout_df["Label_Text"].unique())
    assert test_labels_text == {held_out_label}, (
        f"test_zeroshot.csv cho {held_out_label} chua nhan la: {test_labels_text}"
    )
    heldout_df.to_csv(test_zeroshot_path, index=False)

    return {
        "held_out_label": held_out_label,
        "n_train_benign_attack": len(train_df),
        "n_val_binary": len(val_df),
        "n_test_zeroshot": len(heldout_df),
        "train_benign_attack_label_counts": train_df["Label_Text"].value_counts().to_dict(),
        "val_binary_label_counts": val_df["Label_Text"].value_counts().to_dict(),
        "checksums": {
            "train_benign_attack.csv": file_checksum(train_benign_attack_path),
            "val_binary.csv": file_checksum(val_binary_path),
            "test_zeroshot.csv": file_checksum(test_zeroshot_path),
        },
    }


def main() -> None:
    label_map, attack_labels, benign_label = load_label_config()
    print(f"{len(attack_labels)} attack label se duoc dung lam held-out (leave-one-out): {attack_labels}")
    print(f"Sampling cap: Benign<= {BENIGN_CAP:,}, moi attack class <= {ATTACK_CAP_PER_CLASS:,}")

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "val_fraction": VAL_FRACTION,
        "random_seed": RANDOM_SEED,
        "benign_cap": BENIGN_CAP,
        "attack_cap_per_class": ATTACK_CAP_PER_CLASS,
        "source": str(CLEAN_BY_LABEL_DIR),
        "raw_row_counts": {},
        "splits": {},
        "warnings": [],
    }

    print(f"\nDang doc {benign_label}.parquet ...")
    benign_df, n_before, n_after = load_and_cap_label_parquet(benign_label, is_benign=True, cap=BENIGN_CAP)
    manifest["raw_row_counts"][benign_label] = {"before_cap": n_before, "after_cap": n_after}
    print(f"  before_cap={n_before:,}  after_cap={n_after:,}")

    TRAIN_BENIGN_ONLY_PATH.parent.mkdir(parents=True, exist_ok=True)
    benign_df.to_csv(TRAIN_BENIGN_ONLY_PATH, index=False)
    print(f"  Da ghi {TRAIN_BENIGN_ONLY_PATH} (dung 1 lan duy nhat cho AE-Only/SVDD-Only).")

    print("\nDang doc tat ca file attack ...")
    all_attack_dfs: dict[str, pd.DataFrame] = {}
    for lbl in attack_labels:
        df, n_before, n_after = load_and_cap_label_parquet(lbl, is_benign=False, cap=ATTACK_CAP_PER_CLASS)
        all_attack_dfs[lbl] = df
        manifest["raw_row_counts"][lbl] = {"before_cap": n_before, "after_cap": n_after}
        print(f"  {lbl:30s} before_cap={n_before:>10,}  after_cap={n_after:>8,}")

    for held_out_label in attack_labels:
        print(f"\n=== Dung split: held-out = {held_out_label} ===")
        other_attack_dfs = {lbl: df for lbl, df in all_attack_dfs.items() if lbl != held_out_label}
        heldout_df = all_attack_dfs[held_out_label]

        split_info = build_split_for_label(held_out_label, benign_df, other_attack_dfs, heldout_df)
        manifest["splits"][held_out_label] = split_info

        n_test = split_info["n_test_zeroshot"]
        print(
            f"  train_benign_attack={split_info['n_train_benign_attack']:,}  "
            f"val_binary={split_info['n_val_binary']:,}  "
            f"test_zeroshot={n_test:,}"
        )
        if n_test < 50:
            warning = f"{held_out_label}: test_zeroshot chi co {n_test} dong (< 50) — do tin cay thap."
            manifest["warnings"].append(warning)
            print(f"  CANH BAO: {warning}")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nDa ghi split manifest: {MANIFEST_PATH}")

    if manifest["warnings"]:
        print(f"\n{len(manifest['warnings'])} canh bao ve kich thuoc test set nho — xem split_manifest.json.")
    print(f"\nHoan tat: {len(attack_labels)} split da duoc tao trong {BY_LABEL_DIR}")


if __name__ == "__main__":
    main()
