# UTS-URA: Toward Zero-Shot-Capable Network Intrusion Detection

Bản snapshot code cho phần **Machine Learning (Layer 1)** và **SHAP-based Retrieval (RAG)** của
đồ án. Không bao gồm data thô lớn (`.csv` trong `binary_split_dataset_z_Infilteration/` bị loại
qua `.gitignore` vì vượt giới hạn GitHub) — chỉ giữ code, model checkpoint nhỏ, và kết quả đã chạy.

## Cấu trúc

### `UTS_Layer1/` — Layer 1 ML (binary anomaly gate)

Kiến trúc gốc: 1 split cố định (`binary_split_dataset_z_Infilteration/`), chỉ label
**Infiltration** làm zero-shot held-out (chưa phải leave-one-out 15-label mới).

- `AE_XGBoost/` — Notebook Autoencoder + XGBoost-augmented (`autoencoder_xgboost_ids_v2.ipynb`),
  checkpoint (`checkpoints/`), biểu đồ (`images/`).
- `SVDD_XGBoost/` — Notebook Mixture Deep SVDD + XGBoost-augmented (`svdd_xgboost_ids.ipynb`),
  checkpoint, biểu đồ.
- `XGBoost_only/` — Model + scaler baseline XGBoost thuần (không augment).
- `clad/` — CLAD (Contrastive Learning for Anomaly Detection), reference implementation dùng
  làm baseline so sánh (`model.py`, `train.py`, `eval.py`, `losses/`, `util/`, `checkpoints/`).
- `comparison/` — So sánh tổng hợp giữa các model.
- `binary_split_dataset_z_Infilteration/` — Split data gốc (CSV bị `.gitignore`, chỉ giữ cấu
  trúc thư mục).

### `RAG_SHAP/` — SHAP-Centroid Retrieval (RAG)

Cơ chế retrieval dùng SHAP attribution vector làm query key thay vì text embedding — so sánh
retrieval quality giữa 3 biến thể Layer 1.

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
