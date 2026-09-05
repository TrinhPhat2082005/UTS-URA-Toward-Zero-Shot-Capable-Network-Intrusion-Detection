"""
Computes 4 metrics for the VLM confidence-gate experiment from results.jsonl
(downloaded from the Kaggle run):

  - FP-corrected rate: % of FP-set flows the VLM downgraded to FALSE_POSITIVE.
  - TP-preserved rate: % of TP-set flows kept as CONFIRMED_ATTACK.
  - Net FP reduction: FP rate before vs. after applying the gate.
  - Recall preserved: TP-preserved rate, used as the recall-impact proxy.

Usage:
    python experiments/vlm_gate/analyze_results.py
"""
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "artifacts", "results.jsonl")


def main():
    rows = []
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    fp_rows = [r for r in rows if r["set_type"] == "fp"]
    tp_rows = [r for r in rows if r["set_type"] == "tp"]

    print(f"Loaded {len(rows)} rows ({len(fp_rows)} FP, {len(tp_rows)} TP)\n")

    # Parse-failure accounting (fail-safe: verdict=None -> gate keeps original decision)
    fp_parse_fail = sum(1 for r in fp_rows if r["verdict"] is None)
    tp_parse_fail = sum(1 for r in tp_rows if r["verdict"] is None)
    print(f"Parse failures: {fp_parse_fail}/{len(fp_rows)} FP, {tp_parse_fail}/{len(tp_rows)} TP\n")

    # Metric 1: FP-corrected rate
    fp_corrected = sum(1 for r in fp_rows if r["verdict"] == "FALSE_POSITIVE")
    fp_corrected_rate = fp_corrected / len(fp_rows) if fp_rows else 0.0

    # Metric 2: TP-preserved rate
    tp_preserved = sum(1 for r in tp_rows if r["verdict"] == "CONFIRMED_ATTACK")
    tp_preserved_rate = tp_preserved / len(tp_rows) if tp_rows else 0.0
    tp_wrongly_downgraded = sum(1 for r in tp_rows if r["verdict"] == "FALSE_POSITIVE")

    # Metric 3: Net FP reduction (fail-safe: parse failure keeps original Layer-1 verdict)
    fp_before = len(fp_rows)
    fp_after = sum(1 for r in fp_rows if r["verdict"] != "FALSE_POSITIVE")
    fp_reduction = (fp_before - fp_after) / fp_before if fp_before else 0.0

    # Metric 4: Recall preserved (proxy = TP-preserved rate, the only way the
    # gate could hurt recall — it never touches Layer-1-benign flows)

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"1. FP-corrected rate:     {fp_corrected}/{len(fp_rows)} ({fp_corrected_rate:.1%})")
    print(f"2. TP-preserved rate:     {tp_preserved}/{len(tp_rows)} ({tp_preserved_rate:.1%})")
    if tp_rows:
        print(f"   TP wrongly downgraded: {tp_wrongly_downgraded}/{len(tp_rows)} ({tp_wrongly_downgraded/len(tp_rows):.1%})")
    print(f"3. Net FP reduction:      {fp_before} -> {fp_after} ({fp_reduction:.1%} reduction)")
    print(f"4. Recall preserved:      {tp_preserved_rate:.1%} (proxy via TP-preserved rate)")

    # Confidence distribution for the corrected FPs (sanity check)
    corrected_confidences = [r["confidence"] for r in fp_rows
                              if r["verdict"] == "FALSE_POSITIVE" and r["confidence"] is not None]
    if corrected_confidences:
        avg_conf = sum(corrected_confidences) / len(corrected_confidences)
        print(f"\nAvg confidence on corrected FPs: {avg_conf:.2f}")

    latencies = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        print(f"Avg VLM latency per flow: {avg_lat:.1f}s (n={len(latencies)})")

    out = {
        "n_fp": len(fp_rows),
        "n_tp": len(tp_rows),
        "fp_parse_fail": fp_parse_fail,
        "tp_parse_fail": tp_parse_fail,
        "fp_corrected_rate": fp_corrected_rate,
        "tp_preserved_rate": tp_preserved_rate,
        "tp_wrongly_downgraded_rate": tp_wrongly_downgraded / len(tp_rows) if tp_rows else None,
        "net_fp_reduction": fp_reduction,
        "fp_before": fp_before,
        "fp_after": fp_after,
    }
    out_path = os.path.join(SCRIPT_DIR, "artifacts", "gate_metrics_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
