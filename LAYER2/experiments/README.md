# Thí nghiệm của paper — tra từ bảng sang code

Mỗi thư mục con ứng với một câu hỏi nghiên cứu (RQ) của Section 4–5. Script đánh số
chạy theo thứ tự; file trung gian và kết quả ghi vào `output/<thư mục cùng tên>/`.
Đường dẫn dữ liệu và checkpoint lấy từ `paths.py` (biến môi trường `UTS_*`).

| Bảng / số liệu trong paper | Script | Output |
|---|---|---|
| `tab:res-known`, `tab:res-zs`, `tab:res-zs-type`, pAUC (RQ1, RQ2) | `rq1_rq2_layer1/01_score_val_test.py` → `02_score_train_benign.py` → `03_score_oneclass.py` → `04_layer1_tables.py` | `rq1_rq2_layer1/layer1_tables.md` |
| Bảng so sánh Layer 1 với các baseline mở rộng (RQ1, RQ2), biểu đồ ZS-Acc theo α, latency (slide 6/10) | `rq1_rq2_baselines/01_score_baselines.py` → `02_verify_reproduction.py` → `03_baseline_tables.py --rule meanfold --alphas 0.001 0.002 0.003 0.004 0.005` → `04_baseline_latency.py` → `06_final_results.py` → `05_plot_zs_vs_alpha.py`; lựa chọn checkpoint và ngưỡng ở `final_selection.py` | `rq1_rq2_baselines/final/` (`table_main.csv`, `curve.csv`, `thresholds_main.csv`, `manifest.json`, `zs_vs_alpha.pdf`), `verify.md`, `baseline_latency.json` |
| Cột latency của `tab:res-known`, `tab:res-ident`; `tab:res-latency` (RQ5) | `rq5_latency/latency_detection_path.py` | `rq5_latency/latency.json` |
| `tab:res-ident` (RQ3a) | `rq3_retrieval/rq3a_identify.py <variant>` (chạy 2 lần) | `rq3a_identify/rq3a_<variant>.json` |
| `tab:res-novelty`, tỉ lệ end-to-end zero-day (RQ3b) | `rq3_retrieval/rq3b_novelty_build.py` → `rq3b_novelty_score.py` (cần output của RQ1/RQ2 bước 02, 03) | `rq3b_novelty/R3_full_tables.md` |
| Ngưỡng τ_zd của pipeline (Eq. `zeroday-flag`) | `rq3_retrieval/02_calibrate_novelty.py` | khoá `novelty` trong `artifacts/shap_pool_<variant>.meta.json` |
| Tập tham chiếu R từng fold (Setup 2) | `rq3_retrieval/01_build_loao_pools.py` | `artifacts/loao_pools/<fold>/` |
| `tab:res-pilot` (RQ4) | `run_layer2.ipynb` (dùng `pipeline.py`) + `metrics.py` | `rq4_vlm/pilot_results_ae_xgboost_neutral_v1.jsonl`, `rq4_vlm/PILOT_RESULTS.md` |
| `tab:res-vlm-zeroday` (RQ4, Setup 2) | `rq3_retrieval/02_calibrate_novelty.py --protocol A` → `rq4_vlm/setup2_build_evidence.py` (local) → `rq4_vlm/setup2_vlm.ipynb` (Kaggle, VLM) → `rq4_vlm/setup2_score.py` | `rq4_vlm/setup2_bundle/` (evidence), `setup2_out/` trên Kaggle: `results.jsonl`, `run_meta.json`, `prompt_template.txt`, `scored/` |
| `tab:res-ablation` (RQ4, Setup 1) | chưa viết | `rq4_vlm/` |

## Thành phần pipeline mà các thí nghiệm dùng

| Paper | Code |
|---|---|
| Layer 1 Protocol A (14 fold LOAO), Section 3.3 | `zeroshot_benchmark/04_training_v2/train_*.py` (repo: `LAYER1/`) |
| Layer 1 Protocol B, ngưỡng τ_α (Eq. `tau-alpha`) | `LAYER2/train_layer1.py` |
| Tập tham chiếu R Protocol B | `LAYER2/build_shap_pool.py` |
| SHAP-KNN Retriever (Eq. `type-dist`, `novelty`, `zeroday-flag`) | `LAYER2/shap_knn_retriever.py` |
| Báo cáo VLM, schema 6 mục (Fig. `vlmio`) | `LAYER2/pipeline.py`, `LAYER2/prompts.py` |

## Biến môi trường

| Biến | Mặc định (workspace) | Ý nghĩa |
|---|---|---|
| `UTS_BENCH` | `<workspace>/zeroshot_benchmark` | split `03_splits_v2` và checkpoint Protocol A |
| `UTS_LAYER1_CODE` | `$UTS_BENCH/04_training_v2` | `models/`, `common/` của Layer 1 (repo: `UTS_URA/LAYER1`) |
| `UTS_LAYER1_ART` | `$UTS_BENCH/04_training_v2/artifacts` | checkpoint 14 fold |
| `UTS_DATA` | `LAYER2/` | `data/` và `artifacts/` của Protocol B |
| `UTS_BASELINE_ART` | `<workspace>/baselines_friends/artifacts` | checkpoint các baseline mở rộng (`rq1_rq2_baselines/`) |
| `UTS_RESULTS` | `LAYER2/output` | nơi ghi kết quả thí nghiệm |

Máy 16 GB RAM: đặt `OPENBLAS_NUM_THREADS=2` và `CUDA_VISIBLE_DEVICES=` khi chạy
`rq3a_identify.py`, `02_calibrate_novelty.py` nếu gặp lỗi cấp phát bộ nhớ.
