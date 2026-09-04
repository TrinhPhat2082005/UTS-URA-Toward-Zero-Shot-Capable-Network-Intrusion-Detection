import time
import os
import sys
import subprocess
import argparse
from datetime import datetime
import ssl

# Windows cert store + OpenSSL 3.5.x fails to parse any cert (ASN1 NOT_ENOUGH_DATA
# on this machine), which breaks sentence_transformers/datasets/aiohttp's default
# SSL context creation. Force certifi's bundle instead of the Windows store.
try:
    import certifi
    def _load_certifi_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
        self.load_verify_locations(certifi.where())
    ssl.SSLContext.load_default_certs = _load_certifi_certs
except ImportError:
    pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "core", "layer1_ml"))
sys.path.append(os.path.join(PROJECT_ROOT, "core", "layer2_vlm"))
sys.path.append(os.path.join(PROJECT_ROOT, "core", "capture"))

# Removed old capture_pcap_and_csv as we now use LiveSniffer from core/capture/sniffer.py

def final_cleanup():
    import glob
    print("[Cleanup] Removing temporary data...")
    
    paths_to_clean = [
        os.path.join(PROJECT_ROOT, "data", "raw_pcap", "*.pcap"),
        os.path.join(PROJECT_ROOT, "data", "processed_csv", "*.csv"),
        os.path.join(PROJECT_ROOT, "data", "vlm_dataset", "shap_img", "*.png"),
        os.path.join(PROJECT_ROOT, "data", "vlm_dataset", "chart", "*.png"),
    ]
    
    count = 0
    for pattern in paths_to_clean:
        files = glob.glob(pattern)
        for f in files:
            try:
                os.remove(f)
                count += 1
            except Exception as e:
                print(f"[Error] Could not remove {os.path.basename(f)}: {e}")
    
    print(f"[Cleanup] Deleted {count} files.")

def main():
    parser = argparse.ArgumentParser(description="AGENTIC VLM REAL-TIME IDS PIPELINE")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=3, 
                        help="1: Sniffer Only | 2: Sniffer + ML/SHAP | 3: Full Pipeline + VLM")
    parser.add_argument("--ip", type=str, default="10.20.2.136", help="Host IP")
    parser.add_argument("--interface", type=str, default="Wi-Fi", help="Network interface")
    parser.add_argument("--duration", type=int, default=10, help="Capture duration (s)")
    parser.add_argument("--layer1_variant", type=str, choices=["xgboost", "ae", "svdd"], default="xgboost",
                        help="Layer 1 gate: 'xgboost' (plain gate), 'ae' (AE-augmented XGBoost), "
                             "'svdd' (XGBoost + Mixture Deep SVDD OR-Fusion)")
    parser.add_argument("--retriever", type=str, choices=["shap_centroid", "text_rag"], default="shap_centroid",
                        help="RAG bridge: 'shap_centroid' (cosine match against per-label SHAP "
                             "centroids, default — see RAG_ISSUES.md for why this replaced text_rag), "
                             "'text_rag' (legacy FAISS/sentence-embedding retriever)")
    parser.add_argument("--shap_centroid_threshold", type=float, default=0.8,
                        help="Cosine similarity threshold for shap_centroid retriever. Below this "
                             "for every label -> flagged as a potential zero-day.")

    args = parser.parse_args()
    
    print(f"STARTING AGENTIC IDS - MODE {args.mode}")
    print(f"Interface: {args.interface} | Host IP: {args.ip}")

    # Lazy import to handle potential Windows DLL issues
    l1_handler = None
    vlm = None
    
    # Initialize Sniffer for all modes
    try:
        from sniffer import LiveSniffer
        sniffer = LiveSniffer(interface=args.interface, keep_pcap=True) 
    except Exception as e:
        print(f"[Error] Failed to load Sniffer: {e}")
        return

    if args.mode >= 2:
        print("[1] Loading Layer 1 (ML Core)...")
        try:
            from model_handler import Layer1Inference
            l1_handler = Layer1Inference(
                my_ip=args.ip,
                variant=args.layer1_variant
            )
        except Exception as e:
            print(f"[Error] Failed to load Layer 1: {e}")
            return
            
    rag = None
    if args.mode == 3:
        print("[2] Loading Layer 2 (VLM Forensic Analyst)...")
        try:
            from inference import VLMAnalyst
            vlm = VLMAnalyst()
        except Exception as e:
            print(f"[Error] Failed to load Layer 2: {e}")
            return
        
        print(f"[3] Loading RAG bridge (retriever={args.retriever})...")
        try:
            if args.retriever == "shap_centroid":
                from shap_centroid_retriever import SHAPCentroidRetriever
                rag = SHAPCentroidRetriever(
                    centroids_path=os.path.join(PROJECT_ROOT, "core", "knowledge_base", "embeddings", "shap_centroids.json"),
                    profiles_dir=os.path.join(PROJECT_ROOT, "core", "knowledge_base", "documents", "attack_profiles"),
                    threshold=args.shap_centroid_threshold,
                )
            else:
                from rag_retriever import RAGRetriever
                rag = RAGRetriever(
                    index_path=os.path.join(PROJECT_ROOT, "core", "knowledge_base", "embeddings", "faiss.index"),
                    docs_path=os.path.join(PROJECT_ROOT, "core", "knowledge_base", "embeddings", "documents.json")
                )
        except Exception as e:
            print(f"[Warning] RAG unavailable: {e}. Continuing without RAG.")
            rag = None
    
    print("System ready. Starting analysis loop...")
    
    try:
        # Use Sniffer generator for the loop
        for pcap_file, csv_file in sniffer.start_capture(window_duration=args.duration):
            if not csv_file or not os.path.exists(csv_file):
                continue
            
            if args.mode == 1:
                print(f"[Mode 1] Captured PCAP/CSV via CICFlowMeter. Continuing...")
                continue

            print(f"[ML] Analyzing features...")
            ids_data = l1_handler.process_cicflowmeter_csv(csv_file)

            if ids_data.empty:
                print(f"[ML] Insufficient flow data.")
                if os.path.exists(csv_file): os.remove(csv_file)
                continue

            predictions, _ = l1_handler.run_inference(ids_data)
            anomaly_mask = predictions != 0
            anomaly_count = anomaly_mask.sum()

            if anomaly_count > 0:
                print(f"ALERT: Detected {anomaly_count} suspicious flows.")
                ids_anomaly = ids_data[anomaly_mask].reset_index(drop=True)
                for i, row in ids_anomaly.iterrows():
                    print(f"  -> Abnormal Flow {i+1}: Src IP = {row['Src IP']} <---> Dst IP = {row['Dst IP']}")

                print("[XAI] Running SHAP Explainer...")
                preds_anomaly = predictions[anomaly_mask]

                from explainer import IDSExplainer
                from visualizer import PcapVisualizer

                explainer = IDSExplainer(l1_handler.bst, l1_handler.model_features)

                # Single TreeExplainer call — returns waterfall PNGs, per-flow SHAP
                # dicts (top-k, for logging), and full per-flow SHAP vectors. The
                # shap_centroid retriever reuses full_vectors directly instead of
                # running its own separate TreeExplainer pass on a second model
                # (see shap_centroid_retriever.py docstring for why the retriever
                # was switched to Layer 1's own binary SHAP space).
                shap_paths, shap_dicts, full_vectors = explainer.explain_and_get_dicts(ids_anomaly, preds_anomaly)

                # RAG Retrieval: one context per anomalous flow
                rag_contexts = []
                if rag and args.retriever == "shap_centroid":
                    print(f"[RAG] SHAP-centroid matching for {len(ids_anomaly)} flow(s) "
                          f"(threshold={args.shap_centroid_threshold})...")
                    rag_results = rag.retrieve_batch(full_vectors, top_k=3)
                    for flow_idx, rag_result in enumerate(rag_results):
                        rag_contexts.append(rag.format_context_for_vlm(rag_result))
                        print(f"[RAG] Flow {flow_idx+1} | Predicted: {rag_result['predicted_label']}")
                        print(f"       Latency: {rag_result['latency_ms']:.1f}ms | Zero-day: {rag_result['is_potential_zero_day']}")
                        for m in rag_result['matches']:
                            print(f"       → {m['doc']['title']} (conf: {m['confidence']:.2f})")
                elif rag:
                    print(f"[RAG] Querying Knowledge Base for {len(shap_dicts)} flow(s)...")
                    for flow_idx, shap_dict in enumerate(shap_dicts):
                        rag_result = rag.retrieve(shap_dict, top_k=3)
                        rag_contexts.append(rag.format_context_for_vlm(rag_result))
                        print(f"[RAG] Flow {flow_idx+1} | Query: {rag_result['query']}")
                        print(f"       Latency: {rag_result['latency_ms']:.1f}ms | Zero-day: {rag_result['is_potential_zero_day']}")
                        for m in rag_result['matches']:
                            print(f"       → {m['doc']['title']} (conf: {m['confidence']:.2f})")

                # Merge all per-flow RAG contexts into one block for the VLM
                rag_context = "\n\n".join(rag_contexts) if rag_contexts else None
                
                print("[Viz] Generating network charts...")
                viz = PcapVisualizer(pcap_path=pcap_file, output_dir=os.path.join(PROJECT_ROOT, "data", "vlm_dataset", "chart"))
                chart_paths = viz.generate_all()
                
                if args.mode == 3 and vlm:
                    print("[VLM] Forensic reasoning in progress...")
                    result = vlm.final_reasoning(
                        shap_paths=shap_paths,
                        chart_paths=chart_paths,
                        rag_context=rag_context
                    )
                    
                    print("\nVLM FORENSIC REPORT (RAG-AUGMENTED)")
                    print(result + "\n")
                else:
                    print(f"Layer 1 and SHAP complete. Skipping VLM analysis.")
            else:
                print("Traffic benign.")
                
            # if os.path.exists(csv_file): os.remove(csv_file)
            
    except KeyboardInterrupt:
        print("\n[Terminated] System stopped.")
        # final_cleanup()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Error] Pipeline failure: {e}")

if __name__ == "__main__":
    main()
