# UTS-URA: Toward Zero-Shot-Capable Network Intrusion Detection

Repo này tách rõ 2 tầng khác nhau về bản chất:

- **`pipeline/`** — code sản xuất thật, hệ thống end-to-end chạy được (capture → Layer 1 ML →
  SHAP → RAG → Layer 2 VLM). Đây là những gì `main.py` thực sự `import`/gọi lúc runtime.
- **`experiments/`** — code + kết quả thí nghiệm dùng để *tạo ra* các thành phần trong
  `pipeline/` (notebook train AE/SVDD/CLAD, script build SHAP centroid, so sánh biến thể...).
  Không tự chạy độc lập thành hệ thống, mà là bằng chứng nghiên cứu/quá trình đi tới quyết định
  cuối cùng trong `pipeline/`.

Không bao gồm data thô lớn (`.csv` trong `binary_split_dataset_z_Infilteration/` bị loại qua
`.gitignore` vì vượt giới hạn GitHub) — chỉ giữ code, model checkpoint nhỏ, và kết quả đã chạy.

## `pipeline/` — Hệ thống production

```
pipeline/
├── main.py                    # Orchestrator, 3 mode (capture-only / +ML+SHAP / +RAG+VLM)
├── requirements.txt
├── test_ml_shap_rag.py        # Test nhanh Layer 1 + SHAP + RAG (không cần VLM)
└── core/
    ├── capture/sniffer.py     # TShark → CICFlowMeter → CSV
    ├── layer1_ml/
    │   ├── model_handler.py   # Layer1Inference — load model, map cột CICFlowMeter
    │   ├── explainer.py       # IDSExplainer — SHAP TreeExplainer, get_shap_dict()
    │   └── artifacts/         # Model + scaler cho 3 biến thể: ae/, svdd/, xgboost_only/,
    │                           # + shap_centroid/ (multi-class model cho retrieval)
    ├── knowledge_base/
    │   ├── kb_builder.py      # Build FAISS index từ documents/
    │   ├── documents/         # attack_profiles/, patterns/, sematic/ — 15 attack type
    │   └── embeddings/        # faiss.index, documents.json, shap_centroids.json
    ├── layer2_vlm/
    │   ├── inference.py             # VLMAnalyst — Qwen2.5-VL 7B, RAG-augmented prompting
    │   ├── shap_centroid_retriever.py  # Retriever mặc định (SHAP vector làm query key)
    │   ├── rag_retriever.py            # Retriever legacy (text embedding), baseline so sánh
    │   ├── visualizer.py              # 4 biểu đồ traffic từ PCAP
    │   └── prompter.py
    └── utils/                  # config.py, clear_buffer.py
```

**Chạy thử**: xem hướng dẫn trong `pipeline/main.py` (3 mode: capture-only, +ML+SHAP, đầy đủ
+RAG+VLM) — cần `tshark`, CICFlowMeter, và môi trường có đủ dependency trong `requirements.txt`.

## `experiments/` — Thí nghiệm & bằng chứng nghiên cứu

### `experiments/layer1_ml/` — Layer 1 ML (binary anomaly gate)

Kiến trúc gốc: 1 split cố định (`binary_split_dataset_z_Infilteration/`), chỉ label
**Infiltration** làm zero-shot held-out (chưa phải leave-one-out 15-label mới).

- `ae_xgboost/` — Notebook Autoencoder + XGBoost-augmented (`autoencoder_xgboost_ids_v2.ipynb`),
  checkpoint (`checkpoints/`), biểu đồ (`images/`).
- `svdd_xgboost/` — Notebook Mixture Deep SVDD + XGBoost-augmented (`svdd_xgboost_ids.ipynb`),
  checkpoint, biểu đồ.
- `xgboost_only/` — Model + scaler baseline XGBoost thuần (không augment).
- `clad_baseline/` — CLAD (Contrastive Learning for Anomaly Detection) — **chỉ dùng làm baseline
  so sánh, không merge vào `pipeline/`** (`model.py`, `train.py`, `eval.py`, `losses/`, `util/`,
  `checkpoints/`).
- `comparison/` — So sánh tổng hợp giữa các model (AE/SVDD/CLAD/XGBoost).
- `binary_split_dataset_z_Infilteration/` — Split data gốc (CSV bị `.gitignore`, chỉ giữ cấu
  trúc thư mục).

### `experiments/rag_shap/` — SHAP-Centroid Retrieval (RAG)

Cơ chế retrieval dùng SHAP attribution vector làm query key thay vì text embedding — so sánh
retrieval quality giữa 3 biến thể Layer 1. Đây là nguồn gốc của
`pipeline/core/layer2_vlm/shap_centroid_retriever.py`.

- `kaggle_build_all_variants_shap.py` — Chạy trên Kaggle GPU: tính SHAP vector cho mọi flow ở cả
  3 biến thể (xgboost-only / AE-augmented / SVDD-augmented), lưu ma trận SHAP thô + centroid.
- `eval_retrieval_no_infilt.py` — Đọc lại SHAP vector đã lưu, tính Top-1/3/5 retrieval accuracy
  (2 phương pháp: centroid-mean và KNN), loại nhãn Infiltration (zero-shot, không phải target
  công bằng để so sánh).
- `model/` — Checkpoint Layer 1 đã tải về từ Kaggle, dùng làm input cho
  `kaggle_build_all_variants_shap.py`:
  - `xgboost_only/` — `xgb_only_model_v1_infilteration.pkl`, scaler tương ứng.
  - `ae/` — `ae_best.pt`, `xgb_aug_model.pkl`, scaler tương ứng.
  - `svdd/` — 10 sphere encoder (`encoder_0..9.pt`), `centers.npy`, `radii.npy`,
    `xgb_svdd_model.pkl`, scaler tương ứng.
- `artifacts/all_variants_shap/` — Kết quả đã chạy: SHAP vector đầy đủ (`shap_full_*.npz`),
  centroid từng biến thể (`shap_centroids_*.json`), bảng per-label (`per_label_*.csv`), tổng hợp
  (`summary_*.json`, `cross_variant_summary.json`).
