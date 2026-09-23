"""
Merges the 14 per-fold train_benign_attack.csv files (03_splits_v2/by_label/)
back into ONE full-label CSV with all 14 attack types + Benign present, for
use as build_shap_reference.py's DATA_CSV.

Each fold's train_benign_attack.csv holds the same 560,000 Benign rows and
the same 13 other attack types (byte-identical across folds) minus the one
label held out for that fold's zero-shot round. So the merge is just: take
one fold as the base (13 attacks + Benign), then add back its missing label
from any other fold where that label is present.
"""
import os

import pandas as pd

SPLITS_DIR = r"D:/UTS_RS/workspace/zeroshot_benchmark/03_splits_v2/by_label"
OUT_PATH = r"D:/UTS_RS/workspace/UTS_URA/project_root/data/train_benign_attack_full.csv"

ALL_LABELS = [
    "Bot", "Brute_Force_Web", "Brute_Force_XSS", "DDOS_attack_HOIC",
    "DDOS_attack_LOIC_UDP", "DDoS_attacks_LOIC_HTTP", "DoS_attacks_GoldenEye",
    "DoS_attacks_Hulk", "DoS_attacks_SlowHTTPTest", "DoS_attacks_Slowloris",
    "FTP_BruteForce", "Infiltration", "SQL_Injection", "SSH_Bruteforce",
]


def main():
    base_fold = ALL_LABELS[0]  # any fold works; its own held-out label is added back below
    base_path = os.path.join(SPLITS_DIR, base_fold, "train_benign_attack.csv")
    df = pd.read_csv(base_path)
    print(f"Base ({base_fold}): {len(df)} rows, labels={sorted(df['Label_Text'].unique())}")

    missing_label = base_fold  # base fold's train_benign_attack.csv lacks exactly this label
    donor_fold = next(l for l in ALL_LABELS if l != missing_label)
    donor_path = os.path.join(SPLITS_DIR, donor_fold, "train_benign_attack.csv")
    donor_df = pd.read_csv(donor_path)
    missing_rows = donor_df[donor_df["Label_Text"] == missing_label]
    print(f"Adding back '{missing_label}': {len(missing_rows)} rows (from {donor_fold} fold)")

    full_df = pd.concat([df, missing_rows], ignore_index=True)

    counts = full_df["Label_Text"].value_counts()
    print(f"\nFinal: {len(full_df)} rows, {len(counts)} labels")
    print(counts)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    full_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
