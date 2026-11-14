"""
Demo live cua pipeline trong paper: TShark -> CICFlowMeter -> Layer 1 -> TreeSHAP -> tra tap R (vote 5-NN,
co "khong co tien le gan") -> knowledge base -> VLM. Moi buoc tu Layer 1 tro di la code cua LAYER2/
(cung checkpoint, cung tap R, cung KB va prompt voi notebook vlm_run.ipynb); main.py chi bat goi tin va
goi pipeline cho tung alert.

  --mode 1  chi bat goi tin + CICFlowMeter
  --mode 2  + Layer 1 + TreeSHAP + tra cuu + chon doan KB (khong ve anh, khong VLM)
  --mode 3  + 5 anh + prompt + bao cao VLM cho tung alert (Qwen2.5-VL 7B); ghi ra <UTS_OUT>/live/<thoi diem>/

LAYER2/ mac dinh nam canh project_root/ (UTS_URA/LAYER2); dat UTS_LAYER2 neu khac.
Du lieu/artifacts cua LAYER2 doc theo UTS_DATA (xem LAYER2/config.py).

Luu y: flow live do CICFlowMeter ban Python (core/capture) trich xuat, khong phai CICFlowMeter-V3 2018 da sinh
CSE-CIC-IDS2018. Dinh nghia feature trong KB (vd cot co chi lay goi dau, thu tu HashMap) mo ta ban 2018, nen
co the khong dung voi flow live.
"""
import argparse
import os
import ssl
import sys
import time

# Windows cert store + OpenSSL 3.5.x fails to parse any cert (ASN1 NOT_ENOUGH_DATA
# on this machine), which breaks huggingface/aiohttp's default SSL context
# creation. Force certifi's bundle instead of the Windows store.
try:
    import certifi

    def _load_certifi_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
        self.load_verify_locations(certifi.where())
    ssl.SSLContext.load_default_certs = _load_certifi_certs
except ImportError:
    pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LAYER2_DIR = os.environ.get("UTS_LAYER2", os.path.join(PROJECT_ROOT, "..", "LAYER2"))
sys.path.append(os.path.join(PROJECT_ROOT, "core", "capture"))


def load_layer2(variant, with_vlm):
    """Nap Layer 1, tap R, KB (va VLM neu can) tu LAYER2. Import tre de mode 1 khong phai nap torch/shap
    (tranh loi DLL tren Windows)."""
    sys.path.insert(0, os.path.abspath(LAYER2_DIR))
    import config
    import pipeline
    from evidence.build import EvidenceBuilder
    from gate.engine import Layer1Engine
    from retrieval import reference_set
    from retrieval.store import ReferenceStore

    eb = EvidenceBuilder(l1=Layer1Engine(variant),
                         store=ReferenceStore.load(reference_set.protocol_b_path(variant),
                                                   exclude_labels=config.RAG_EXCLUDE_LABELS))
    vlm = None
    if with_vlm:
        from vlm.runner import VLMRunner
        vlm = VLMRunner()
    return config, pipeline, eb, vlm


def main():
    parser = argparse.ArgumentParser(description="AGENTIC VLM REAL-TIME IDS PIPELINE")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=3,
                        help="1: Sniffer only | 2: + Layer 1, SHAP, lookup, KB | 3: + VLM report")
    parser.add_argument("--ip", type=str, default="10.20.2.136", help="Host IP (chi de in log)")
    parser.add_argument("--interface", type=str, default="Wi-Fi", help="Network interface")
    parser.add_argument("--duration", type=int, default=10, help="Capture window (s)")
    parser.add_argument("--variant", type=str, choices=["ae_xgboost", "svdd_xgboost"],
                        default="ae_xgboost", help="Layer 1 gate (paper: AE-XGBoost)")
    args = parser.parse_args()

    print(f"STARTING AGENTIC IDS - MODE {args.mode}")
    print(f"Interface: {args.interface} | Host IP: {args.ip}")

    try:
        from sniffer import LiveSniffer
        sniffer = LiveSniffer(interface=args.interface, keep_pcap=True)
    except Exception as e:
        print(f"[Error] Failed to load Sniffer: {e}")
        return

    config = pipeline = eb = vlm = None
    run_dir = None
    if args.mode >= 2:
        print(f"[1] Loading Layer 1 + reference set + KB from LAYER2 (variant={args.variant})...")
        try:
            config, pipeline, eb, vlm = load_layer2(args.variant, with_vlm=args.mode == 3)
        except Exception as e:
            print(f"[Error] Failed to load LAYER2 components: {e}")
            return
        if args.mode == 3:
            run_dir = os.path.join(config.OUT_DIR, "live", time.strftime("%Y%m%d_%H%M%S"))
            pipeline.write_run_meta(run_dir, eb, vlm, extra={"source": "live capture", "interface": args.interface})
            print(f"    reports -> {run_dir}")
    from flow_columns import load_cicflowmeter_csv

    print("System ready. Starting analysis loop...")
    n_alert = 0
    try:
        for pcap_file, csv_file in sniffer.start_capture(window_duration=args.duration):
            if not csv_file or not os.path.exists(csv_file):
                continue
            if args.mode == 1:
                print("[Mode 1] Captured PCAP/CSV via CICFlowMeter. Continuing...")
                continue

            l1 = eb.l1
            base = eb.base
            flows = load_cicflowmeter_csv(csv_file, base)
            if flows.empty:
                print("[ML] Insufficient flow data.")
                continue

            proba = l1.predict_proba(flows[base].to_numpy("float64"))
            alerts = flows[proba >= l1.threshold].reset_index(drop=True)
            if alerts.empty:
                print(f"[ML] {len(flows)} flow, traffic benign.")
                continue
            print(f"ALERT: {len(alerts)} / {len(flows)} flow vuot nguong {l1.threshold:.4f}.")

            for i, row in alerts.iterrows():
                n_alert += 1
                src, dst = row.get("Src IP", "?"), row.get("Dst IP", "?")
                if args.mode == 2:
                    from utils.timing import StageTimer
                    t = StageTimer()
                    ev = eb.build(row, out_dir=None, timer=t, with_images=False)
                    if not ev["alert"]:
                        continue
                    look = ev["lookup"]
                    print(f"  [{i + 1}] {src} -> {dst} | p={ev['p']:.3f} | nearest label {look['label']} "
                          f"(votes {look['votes']}) | s={look['novelty']:.4f} flag={look['flagged']} | "
                          f"techniques {ev['kb']['techniques'] or '-'} | {len(ev['kb']['passages'])} KB passages | "
                          + " ".join(f"{k} {v * 1000:.1f}ms" for k, v in t.as_dict().items()))
                    continue

                row = row.copy()
                row["sample_id"] = f"live{n_alert:05d}"
                rec = pipeline.run_sample(row, eb, vlm, run_dir)
                if rec["run_error"]:
                    print(f"  [{i + 1}] {src} -> {dst} | error: {rec['run_error']}")
                    continue
                if not rec.get("alert"):
                    continue
                report = os.path.join(run_dir, rec["sample_id"], "output", "report.md")
                print(f"\n===== VLM FORENSIC REPORT {rec['sample_id']} {src} -> {dst} "
                      f"(nearest label: {rec['pred_label']}; no close precedent: {rec['flagged']}; "
                      f"{rec['latency_total']:.1f} s) =====")
                print(open(report, encoding="utf-8").read() + "\n")
    except KeyboardInterrupt:
        print("\n[Terminated] System stopped.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Error] Pipeline failure: {e}")


if __name__ == "__main__":
    main()
