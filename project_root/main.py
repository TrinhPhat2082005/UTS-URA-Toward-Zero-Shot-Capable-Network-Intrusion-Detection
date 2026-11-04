"""
Demo live cua pipeline trong paper: TShark -> CICFlowMeter -> Layer 1 -> SHAP ->
SHAP-KNN Retriever -> VLM. Moi thanh phan tu Layer 1 tro di la code cua LAYER2/
(cung model, cung tap tham chieu, cung prompt voi cac thi nghiem), main.py chi lo
bat goi tin va noi cac buoc lai voi nhau.

  --mode 1  chi bat goi tin + CICFlowMeter
  --mode 2  + Layer 1 (quyet dinh + TreeSHAP) + retrieval (top-3 ung vien, co zero-day)
  --mode 3  + bao cao VLM cho tung alert (Qwen2.5-VL 7B)

LAYER2/ mac dinh nam canh project_root/ (UTS_URA/LAYER2); dat UTS_LAYER2 neu khac.
Du lieu/artifacts cua LAYER2 doc theo UTS_DATA (xem LAYER2/config.py).
"""
import argparse
import os
import ssl
import sys

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
    """Nap Layer 1, retriever (va VLM neu can) tu LAYER2. Import tre de mode 1
    khong phai nap torch/shap (tranh loi DLL tren Windows)."""
    sys.path.insert(0, os.path.abspath(LAYER2_DIR))
    import config
    import pipeline
    from retrieval import reference_set
    from retrieval.retriever import SHAPKNNRetriever

    l1 = pipeline.Layer1Engine(variant)
    retriever = SHAPKNNRetriever(
        npz_path=str(reference_set.protocol_b_path(variant)),
        profiles_dir=config.ATTACK_PROFILES_DIR,
        tau_zd=config.RAG_ZERODAY_THRESHOLD,
        k=config.RAG_NOVELTY_K,
    )
    ev = pipeline.EvidenceBuilder(retriever)
    vlm = pipeline.VLMRunner() if with_vlm else None
    return pipeline, l1, ev, vlm


def main():
    parser = argparse.ArgumentParser(description="AGENTIC VLM REAL-TIME IDS PIPELINE")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=3,
                        help="1: Sniffer only | 2: + Layer 1, SHAP, retrieval | 3: + VLM report")
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

    pipeline = l1 = ev = vlm = None
    if args.mode >= 2:
        print(f"[1] Loading Layer 1 + retriever from LAYER2 (variant={args.variant})...")
        try:
            pipeline, l1, ev, vlm = load_layer2(args.variant, with_vlm=args.mode == 3)
        except Exception as e:
            print(f"[Error] Failed to load LAYER2 components: {e}")
            return
    from flow_columns import load_cicflowmeter_csv

    print("System ready. Starting analysis loop...")
    try:
        for pcap_file, csv_file in sniffer.start_capture(window_duration=args.duration):
            if not csv_file or not os.path.exists(csv_file):
                continue
            if args.mode == 1:
                print("[Mode 1] Captured PCAP/CSV via CICFlowMeter. Continuing...")
                continue

            base = list(l1.feature_names[:l1.n_base])
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
                src, dst = row.get("Src IP", "?"), row.get("Dst IP", "?")
                if args.mode == 2:
                    _, sv = l1.score(row[base].to_numpy("float64").reshape(1, -1))
                    res, _, t = ev.retrieve(sv[0])
                    cands = ", ".join(f"{m['matched_label']} ({m['distance']:.3f})" for m in res["matches"])
                    print(f"  [{i + 1}] {src} -> {dst} | top-3: {cands} | novelty {res['novelty_score']:.3f}"
                          f" | zero-day: {res['is_potential_zero_day']} | {t * 1000:.1f} ms")
                    continue
                rec = pipeline.analyze_sample(row, l1, ev, vlm, base,
                                              meta={"row_idx": f"live{i}", "src": src, "dst": dst})
                if rec["run_error"]:
                    print(f"  [{i + 1}] {src} -> {dst} | error: {rec['run_error']}")
                    continue
                print(f"\n===== VLM FORENSIC REPORT [{i + 1}] {src} -> {dst} "
                      f"(top-3: {', '.join(rec['rag_candidates'])}; "
                      f"zero-day: {rec['rag_is_potential_zero_day']}) =====")
                print(rec["raw_output"] + "\n")
    except KeyboardInterrupt:
        print("\n[Terminated] System stopped.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Error] Pipeline failure: {e}")


if __name__ == "__main__":
    main()
