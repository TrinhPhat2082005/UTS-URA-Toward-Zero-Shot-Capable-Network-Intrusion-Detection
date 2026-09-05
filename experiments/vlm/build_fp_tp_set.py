import os
import sys
import argparse

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(WORKSPACE_ROOT, "UTS_Layer1", "binary_split_dataset_z_Infilteration")
ARTIFACTS_OUT_DIR = os.path.join(SCRIPT_DIR, "artifacts")

sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "project_root", "core", "layer1_ml"))
from model_handler import Layer1Inference  # noqa: E402


def run_split(handler, csv_path, split_name):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    labels = df["Label"].astype(int).values

    preds, preds_prob = handler.run_inference(df.copy())

    anomaly_mask = preds == 1
    fp_mask = anomaly_mask & (labels == 0)
    tp_mask = anomaly_mask & (labels == 1)

    print(f"[{split_name}] n={len(df)}  flagged_anomaly={anomaly_mask.sum()}  "
          f"FP={fp_mask.sum()}  TP={tp_mask.sum()}")

    fp_rows = df.loc[fp_mask].copy()
    fp_rows["xgb_proba"] = preds_prob[fp_mask]
    fp_rows["source_split"] = split_name

    tp_rows = df.loc[tp_mask].copy()
    tp_rows["xgb_proba"] = preds_prob[tp_mask]
    tp_rows["source_split"] = split_name

    return fp_rows, tp_rows


def main():
    parser = argparse.ArgumentParser(description="Build natural FP/TP sets for VLM gate experiment")
    parser.add_argument("--variant", default="xgboost", choices=["xgboost", "ae", "svdd"])
    args = parser.parse_args()

    os.makedirs(ARTIFACTS_OUT_DIR, exist_ok=True)

    print(f"[Layer 1] Loading variant={args.variant}...")
    handler = Layer1Inference(my_ip="0.0.0.0", variant=args.variant)

    fp_parts, tp_parts = [], []
    for split_name in ["test_binary.csv", "val_binary.csv"]:
        csv_path = os.path.join(DATA_DIR, split_name)
        fp_rows, tp_rows = run_split(handler, csv_path, split_name)
        fp_parts.append(fp_rows)
        tp_parts.append(tp_rows)

    fp_set = pd.concat(fp_parts, ignore_index=True)
    tp_set = pd.concat(tp_parts, ignore_index=True)

    fp_out = os.path.join(ARTIFACTS_OUT_DIR, f"fp_set_{args.variant}.csv")
    tp_out = os.path.join(ARTIFACTS_OUT_DIR, f"tp_set_{args.variant}.csv")
    fp_set.to_csv(fp_out, index=False)
    tp_set.to_csv(tp_out, index=False)

    print("\n" + "=" * 60)
    print(f"Total FP (natural, threshold={handler.threshold}): {len(fp_set)}")
    print(f"Total TP: {len(tp_set)}")
    print(f"Saved: {fp_out}")
    print(f"Saved: {tp_out}")
    if len(fp_set) < 30:
        print("\n[Warning] FP count is small — statistical power for the gate "
              "experiment will be limited. Consider reporting this count before "
              "proceeding to the VLM-run stage.")


if __name__ == "__main__":
    main()
