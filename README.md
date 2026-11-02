# UTS-URA: Toward Zero-Shot-Capable Network Intrusion Detection

Code của paper: Layer 1 (XGBoost tăng cường AE/SVDD, ràng buộc đơn điệu, TreeSHAP chính xác),
SHAP-KNN Retriever trên không gian attribution, và báo cáo VLM có grounding MITRE ATT&CK.
Chỉ chứa code và kết quả nhỏ; dữ liệu, split và checkpoint nằm ngoài repo (xem biến môi trường
bên dưới).

## Cấu trúc

```
UTS_URA/
├── LAYER1/                    Layer 1, Protocol A (14 fold leave-one-attack-out) — Section 3.2–3.3
│   ├── config/labels.json
│   ├── 02_cleaning/           raw CSV → parquet đã làm sạch, theo nhãn
│   ├── 03_splits/             split LOAO dùng trong paper (build_splits.py): benign 700K → 560K train / 140K val
│   └── 04_training_v2/        train XGBoost-only, AE-only, SVDD-only, CLAD, AE-XGBoost, SVDD-XGBoost
│                              (mono, config common/xgb_config_v5_mono.py)
│
├── LAYER2/                    Retrieval + VLM, Protocol B — Section 3.4–3.7
│   ├── train_layer1.py        Layer 1 Protocol B, ngưỡng theo ngân sách FPR
│   ├── build_shap_pool.py     tập tham chiếu R
│   ├── shap_knn_retriever.py  một lần tính L2 → top-3 + cờ zero-day
│   ├── pipeline.py, prompts.py, flow_visualizer.py, metrics.py
│   ├── knowledge_base/        profile MITRE ATT&CK
│   ├── experiments/           thí nghiệm theo RQ — xem experiments/README.md (bảng paper → script)
│   └── output/                kết quả đã chạy, theo RQ
│
└── project_root/              demo live: TShark → CICFlowMeter → các thành phần của LAYER2
    ├── main.py
    └── core/capture/          sniffer.py, flow_columns.py (đổi tên cột CICFlowMeter)
```

## Tra từ paper sang code

`LAYER2/experiments/README.md` liệt kê từng bảng của Section 5 cùng script và file output
sinh ra nó.

## Biến môi trường

| Biến | Ý nghĩa | Ví dụ (máy local) |
|---|---|---|
| `UTS_DATA` | `data/` và `artifacts/` của LAYER2 | `D:\UTS_RS\workspace\LAYER2` |
| `UTS_BENCH` | split và checkpoint Protocol A | `D:\UTS_RS\workspace\zeroshot_benchmark` |
| `UTS_LAYER1_CODE` | `models/`, `common/` của Layer 1 | `<repo>\LAYER1\04_training_v2` |
| `UTS_RESULTS` | nơi ghi kết quả thí nghiệm | mặc định `LAYER2/output` |

## Demo live

```bash
python project_root/main.py --mode 1   # chỉ bắt gói tin + CICFlowMeter
python project_root/main.py --mode 2   # + Layer 1, TreeSHAP, top-3 ứng viên, cờ zero-day
python project_root/main.py --mode 3   # + báo cáo Qwen2.5-VL 7B cho từng alert
```

Cần `tshark`, CICFlowMeter, và (mode 3) Qwen2.5-VL 7B. Mode 2/3 nạp model và tập tham chiếu
từ `UTS_DATA`.
