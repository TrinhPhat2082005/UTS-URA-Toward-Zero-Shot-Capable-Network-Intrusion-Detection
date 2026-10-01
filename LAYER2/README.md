# LAYER2 — SHAP-vector retrieval và báo cáo VLM

Code của Section 3.4–3.7 trong paper, chạy trên Layer 1 Protocol B (một mô hình,
14 loại tấn công). Thí nghiệm của paper nằm trong `experiments/` — xem
`experiments/README.md` để tra từ bảng trong paper sang script.

```
flow → Layer 1 (xác suất + TreeSHAP) → SHAP-KNN Retriever (top-3 + cờ zero-day)
     → 5 ảnh (SHAP waterfall + 4 biểu đồ luồng) → VLM → báo cáo 6 mục
```

## File

| File | Paper | Việc |
|---|---|---|
| `config.py` | | Đường dẫn và tham số; dữ liệu đọc theo biến `UTS_DATA` |
| `train_layer1.py` | 3.3, Eq. `tau-alpha` | Train Layer 1 Protocol B (config v5 mono), ngưỡng theo ngân sách FPR α = 0.4% |
| `build_shap_pool.py` | 3.5 | Tập tham chiếu R: SHAP của mọi flow attack trong train_full |
| `shap_knn_retriever.py` | 3.5, Eq. `type-dist`, `novelty`, `zeroday-flag` | Một lần tính L2 → 3 loại gần nhất + điểm novelty; τ_zd đọc từ `shap_pool_<variant>.meta.json` |
| `pipeline.py` | 3.7 | `Layer1Engine`, `EvidenceBuilder`, `VLMRunner`, `analyze_sample` |
| `prompts.py` | 3.7, Fig. `vlmio` | System prompt, schema báo cáo, parser |
| `flow_visualizer.py` | 3.7 | 4 biểu đồ luồng |
| `metrics.py`, `METRIC.md` | 4.5 | Chấm báo cáo VLM |
| `knowledge_base/attack_profiles/` | 3.6 | 14 profile MITRE ATT&CK + unknown_anomaly |
| `models/` | | Bản sao kiến trúc AE/SVDD của Layer 1 (để LAYER2 tự chạy trên Kaggle) |
| `run_layer2.ipynb`, `_make_notebook.py` | | Notebook chạy VLM trên Kaggle và script sinh lại notebook |

## Chạy

1. `python LAYER2/train_layer1.py` → `artifacts/layer1/<variant>/`
2. `python LAYER2/build_shap_pool.py` → `artifacts/shap_pool_<variant>.npz`
3. `python LAYER2/experiments/rq3_retrieval/02_calibrate_novelty.py` → τ_zd vào file meta
4. `run_layer2.ipynb` (GPU) để sinh báo cáo VLM.

Demo live (bắt gói tin thật) dùng chính các thành phần này qua `project_root/main.py`.

## Còn thiếu so với paper

- `prompts.py` chưa có trường `selected` (một trong 3 ứng viên hoặc `none`) và `tactic`
  mà Section 4.5 mô tả; cần sửa trước khi chạy Setup 1 và Setup 2.
- Script Setup 1 (`tab:res-ablation`) và Setup 2 (`tab:res-vlm-zeroday`) chưa viết.
