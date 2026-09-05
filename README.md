# UTS-URA: Toward Zero-Shot-Capable Network Intrusion Detection

Repo tách rõ 2 tầng khác nhau về bản chất:

- **`project_root/`** : hệ thống end-to-end chạy được (capture → Layer 1 ML
  → SHAP → RAG → Layer 2 VLM). Đây là những gì `main.py` thực sự `import`/gọi lúc runtime.
- **`experiments/`** : code + kết quả thí nghiệm dùng để *tạo ra* các thành phần trong
  `project_root/` (notebook train AE/SVDD/CLAD, script build SHAP centroid/k-NN, so sánh biến
  thể...). Không tự chạy độc lập thành hệ thống, mà là bằng chứng nghiên cứu/quá trình đi tới
  quyết định cuối cùng trong `project_root/`.

Không bao gồm data thô lớn (`.csv` trong `binary_split_dataset_z_Infilteration/` bị loại qua
`.gitignore` vì vượt giới hạn GitHub) — chỉ giữ code, model checkpoint nhỏ, và kết quả đã chạy.

## Sơ đồ cấu trúc

```
UTS_URA/
├── README.md
│
├── project_root/                       # Hệ thống production
│   ├── main.py                         # Orchestrator — 3 mode (capture-only / +ML+SHAP / +RAG+VLM)
│   ├── requirements.txt
│   ├── test_ml_shap_rag.py             # Test nhanh Layer 1 + SHAP + RAG (không cần VLM)
│   └── core/
│       ├── capture/
│       │   └── sniffer.py              # TShark → CICFlowMeter → CSV
│       ├── layer1_ml/
│       │   ├── model_handler.py        # Layer1Inference — load model, map cột CICFlowMeter
│       │   ├── explainer.py            # IDSExplainer — SHAP TreeExplainer
│       │   └── artifacts/              # Model + scaler cho từng biến thể
│       │       ├── xgboost_only/       #   XGBoost thuần (baseline)
│       │       ├── ae/                 #   AE-augmented XGBoost
│       │       ├── svdd/               #   Mixture SVDD-augmented XGBoost
│       │       └── shap_centroid/      #   Model phụ dùng để build SHAP centroid retrieval
│       ├── knowledge_base/
│       │   ├── kb_builder.py           # Build FAISS index từ documents/
│       │   ├── documents/              # attack_profiles/, patterns/, sematic/ — 15 attack type
│       │   └── embeddings/             # faiss.index, documents.json, shap_centroids.json
│       ├── layer2_vlm/
│       │   ├── inference.py            # VLMAnalyst — Qwen2.5-VL 7B, RAG-augmented prompting
│       │   ├── shap_centroid_retriever.py  # Retriever mặc định (SHAP vector làm query key)
│       │   ├── rag_retriever.py        # Retriever legacy (text embedding), baseline so sánh
│       │   └── visualizer.py           # 4 biểu đồ traffic từ PCAP
│       └── utils/
│           ├── config.py
│           └── clear_buffer.py
│
└── experiments/                        # Thí nghiệm & bằng chứng nghiên cứu
    │
    ├── ml/                             # Layer 1 ML — huấn luyện & so sánh model
    │   ├── binary_split_dataset_z_Infilteration/   # Split gốc (CSV bị .gitignore)
    │   ├── xgboost_only/                # Baseline XGBoost thuần
    │   ├── ae_xgboost/                  # Notebook + checkpoint AE + XGBoost-augmented
    │   ├── svdd_xgboost/                # Notebook + checkpoint Mixture Deep SVDD + XGBoost
    │   ├── clad_baseline/               # CLAD — chỉ dùng baseline so sánh, KHÔNG merge vào pipeline
    │   └── comparison/                  # So sánh tổng hợp AE/SVDD/CLAD/XGBoost
    │
    ├── rag_shap/                        # SHAP-vector retrieval (nguồn gốc shap_centroid_retriever.py)
    │   ├── kaggle_build_all_variants_shap.py   # Tính SHAP vector + centroid cho cả 3 biến thể
    │   ├── eval_retrieval_no_infilt.py         # So Top-1/3/5 accuracy: centroid-mean vs k-NN
    │   ├── model/                       # Checkpoint Layer 1 tải từ Kaggle (xgboost_only/ae/svdd)
    │   └── artifacts/all_variants_shap/ # SHAP vector, centroid, summary đã chạy
    │
    └── vlm/                             # Thí nghiệm VLM confidence-gate (FP-correction)
        ├── build_fp_tp_set.py           # Tạo bộ FP/TP tự nhiên từ Layer 1 (test+val split)
        ├── flow_visualizer.py           # 4 chart traffic từ flow CSV (thay PcapVisualizer)
        ├── gate_prompt.py               # Các phiên bản system prompt cho VLM gate
        ├── shap_centroid_retriever.py   # Retriever centroid — bản độc lập cho Kaggle
        ├── shap_knn_retriever.py        # Retriever k-NN (so sánh với centroid)
        ├── prepare_evidence_kaggle_variant_centroid.py  # Build evidence bundle — centroid
        ├── prepare_evidence_kaggle_variant_knn.py       # Build evidence bundle — k-NN
        ├── run_vlm_remote_clean.py      # Chạy VLM trên Kaggle GPU, sinh report + verdict
        ├── analyze_results.py           # Tính FP-corrected / TP-preserved / net FP reduction
        └── artifacts/
            ├── evidence_13_{xgb,ae,svdd}_{centroid,knn}/  # Evidence bundle (SHAP+chart+RAG)
            └── results/{centroid,knn}/  # Output VLM (results_13_{variant}_{centroid,knn}.jsonl)
```

## `project_root/` — Chạy thử

```bash
cd project_root
pip install -r requirements.txt
python main.py --mode 1   # Capture-only (TShark + CICFlowMeter)
python main.py --mode 2   # + Layer 1 ML + SHAP
python main.py --mode 3   # + RAG + Layer 2 VLM (đầy đủ)
```

Cần `tshark`, CICFlowMeter, và (mode 3) Qwen2.5-VL 7B tải sẵn. Trước khi chạy RAG, build FAISS
index một lần: `python core/knowledge_base/kb_builder.py`.

## `experiments/` — Ba nhóm thí nghiệm

| Thư mục | Câu hỏi nghiên cứu | Kết quả nuôi vào production |
|---|---|---|
| `ml/` | AE hay SVDD-augmented XGBoost tốt hơn CLAD/XGBoost thuần cho zero-shot Infiltration? | `project_root/core/layer1_ml/artifacts/{ae,svdd,xgboost_only}/` |
| `rag_shap/` | SHAP vector làm query key cho RAG có đáng tin hơn text embedding không? | `project_root/core/layer2_vlm/shap_centroid_retriever.py` |
| `vlm/` | VLM có thể "sửa" False Positive của Layer 1 mà không làm mất True Positive không? | Chưa merge — đang ở giai đoạn thí nghiệm (xem `PROJECT_STATE.md`) |

Chi tiết từng script trong `experiments/vlm/`: xem comment đầu file — mỗi script đều có docstring
ngắn nêu vai trò trong pipeline 2 giai đoạn (prepare evidence trên Kaggle CPU → chạy VLM trên
Kaggle GPU → phân tích kết quả local).
