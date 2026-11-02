# Thí nghiệm Layer 1 (RQ1, RQ2) — tra từ bảng sang code

Đánh giá Layer 1 trên Protocol A (14 fold leave-one-attack-out). Các script **không train lại**:
chúng nạp checkpoint đã có, chấm điểm, rồi áp ngưỡng theo ngân sách FPR
τ = Q₍₁₋α₎(điểm trên 560K benign train). Ngưỡng in trong các script `04_training_v2/train_*.py`
(F1-max) chỉ để tham khảo, không phải ngưỡng của paper.

Script đánh số chạy theo thứ tự; kết quả ghi vào `LAYER1/output/<thư mục cùng tên>/`.
Đường dẫn lấy từ `paths.py` (biến môi trường `UTS_*`).

| Bảng / số liệu | Script | Output |
|---|---|---|
| `tab:res-known`, `tab:res-zs`, `tab:res-zs-type`, pAUC — 6 model của paper | `rq1_rq2_layer1/01_score_val_test.py` → `02_score_train_benign.py` → `03_score_oneclass.py` → `04_layer1_tables.py` | `rq1_rq2_layer1/layer1_tables.md`, `sweep_*_scores.npz` |
| Bảng so sánh với các baseline mở rộng, biểu đồ ZS-Acc theo α, latency (slide RQ1/RQ2) | `rq1_rq2_baselines/01_score_baselines.py` → `02_verify_reproduction.py` → `03_baseline_tables.py --rule meanfold --alphas 0.001 0.002 0.003 0.004 0.005` → `04_baseline_latency.py` → `06_final_results.py` → `05_plot_zs_vs_alpha.py`; chọn checkpoint và quy tắc ngưỡng ở `final_selection.py` | `rq1_rq2_baselines/final/` (`final_results.md`, `table_main.csv`, `curve.csv`, `thresholds_main.csv`, `manifest.json`, `zs_vs_alpha.pdf`), `verify.md`, `baseline_latency.json` |

RQ3b (`LAYER2/experiments/rq3_retrieval/`) đọc lại `sweep_train_scores.npz`, `sweep_all_scores.npz`
và `scores_svdd_xgboost.npz` ở đây, và import `rq1_rq2_baselines/baselines.py`.

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `UTS_WS` | cha của repo (`workspace/`) | thư mục gốc workspace |
| `UTS_BENCH` | `$UTS_WS/zeroshot_benchmark` | split `03_splits_v2` và checkpoint Protocol A |
| `UTS_LAYER1_CODE` | `LAYER1/04_training_v2` | `models/`, `common/` của Layer 1 |
| `UTS_LAYER1_ART` | `$UTS_BENCH/04_training_v2/artifacts` | checkpoint 14 fold |
| `UTS_BASELINE_ART` | `$UTS_WS/baselines_friends/artifacts` | checkpoint các baseline mở rộng |
| `UTS_L1_RESULTS` | `LAYER1/output` | nơi ghi kết quả |

Chạy bằng conda env `ai_env`. Máy 16 GB RAM: đặt `OPENBLAS_NUM_THREADS=2` và `CUDA_VISIBLE_DEVICES=`
nếu gặp lỗi cấp phát bộ nhớ.
