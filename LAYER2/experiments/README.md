# Thí nghiệm của paper — tra từ bảng sang code

Mỗi thư mục con ứng với một câu hỏi nghiên cứu (RQ) của Section 4–5. Script đánh số
chạy theo thứ tự; file trung gian và kết quả ghi vào `output/<thư mục cùng tên>/`.
Đường dẫn dữ liệu và checkpoint lấy từ `paths.py` (biến môi trường `UTS_*`).

| Bảng / số liệu trong paper | Script | Output |
|---|---|---|
| `tab:res-known`, `tab:res-zs`, `tab:res-zs-type`, pAUC, bảng baseline mở rộng (RQ1, RQ2) | đã chuyển sang `LAYER1/04_experiments/` — xem `LAYER1/04_experiments/README.md` | `LAYER1/output/tables/`, `LAYER1/output/scores/` |
| Cột latency của `tab:res-known`, `tab:res-ident`; `tab:res-latency` (RQ5) | `rq5_latency/latency_detection_path.py` | `rq5_latency/latency.json` |
| `tab:res-ident` (RQ3a) | `rq3_retrieval/rq3a_identify.py <variant>` (chạy 2 lần) | `rq3a_identify/rq3a_<variant>.json` |
| `tab:res-novelty`, tỉ lệ end-to-end zero-day (RQ3b) | `rq3_retrieval/rq3b_novelty_build.py` → `rq3b_novelty_score.py` (cần cache điểm của RQ1/RQ2 trong `LAYER1/output/scores/`) | `rq3b_novelty/R3_full_tables.md` |
| Ngưỡng τ_zd của pipeline (Eq. `zeroday-flag`) | `rq3_retrieval/02_calibrate_novelty.py` | khoá `novelty` trong `artifacts/reference_sets/protocol_b/<variant>.meta.json` |
| Tập tham chiếu R từng fold (Setup 2) | `rq3_retrieval/01_build_loao_pools.py` | `artifacts/reference_sets/protocol_a/<fold>/<variant>.npz` |
| `tab:res-pilot` (RQ4) | `run_layer2.ipynb` (dùng `pipeline.py`) + `metrics.py` | `rq4_vlm/pilot_results_ae_xgboost_neutral_v1.jsonl`, `rq4_vlm/PILOT_RESULTS.md` |
| `tab:res-vlm-zeroday` (RQ4, Setup 2) | `rq3_retrieval/02_calibrate_novelty.py --protocol A` → `rq4_vlm/setup2_build_evidence.py` (local) → `rq4_vlm/setup2_vlm.ipynb` (Kaggle, VLM) → `rq4_vlm/setup2_score.py` | `rq4_vlm/setup2_bundle/` (evidence), `setup2_out/` trên Kaggle: `results.jsonl`, `run_meta.json`, `prompt_template.txt`, `scored/` |
| `tab:res-ablation` (RQ4, Setup 1) | `rq4_vlm/setup1_build_evidence.py` (local) → `rq4_vlm/setup1_vlm.ipynb` (Kaggle, VLM) → `rq4_vlm/setup1_score.py` | `rq4_vlm/setup1_bundle/<variant>/` (evidence), `setup1_out/` trên Kaggle |

## Thành phần pipeline mà các thí nghiệm dùng

| Paper | Code |
|---|---|
| Layer 1 Protocol A (14 fold LOAO), Section 3.3 | `zeroshot_benchmark/04_training_v2/train_*.py` (repo: `LAYER1/03_training/`) |
| Layer 1 Protocol B, ngưỡng τ_α (Eq. `tau-alpha`) | `LAYER2/layer1/train.py` |
| Vector SHAP (cả hai protocol) | `LAYER2/layer1/engine.py` (`Layer1Engine`, `Layer1Engine.from_fold`) |
| Tập tham chiếu R | `LAYER2/retrieval/reference_set.py` (Protocol B), `rq3_retrieval/01_build_loao_pools.py` (Protocol A) |
| SHAP-KNN Retriever (Eq. `type-dist`, `novelty`, `zeroday-flag`) | `LAYER2/retrieval/retriever.py`, `LAYER2/retrieval/novelty.py` |
| Evidence và bundle gửi VLM | `LAYER2/evidence/build.py`, `LAYER2/evidence/bundle.py` |
| Báo cáo VLM (Fig. `vlmio`) | `LAYER2/vlm/runner.py`, `LAYER2/prompts.py` (bật/tắt UNKNOWN: `allow_unknown`) |

## Biến môi trường

| Biến | Mặc định (workspace) | Ý nghĩa |
|---|---|---|
| `UTS_BENCH` | `<workspace>/zeroshot_benchmark` | split `03_splits_v2` và checkpoint Protocol A |
| `UTS_LAYER1_CODE` | `$UTS_BENCH/04_training_v2` | `models/`, `common/` của Layer 1 (repo: `UTS_URA/LAYER1/03_training`) |
| `UTS_LAYER1_ART` | `$UTS_BENCH/04_training_v2/artifacts` | checkpoint 14 fold |
| `UTS_DATA` | `LAYER2/` | `data/` và `artifacts/` của Protocol B |
| `UTS_BASELINE_ART` | `<workspace>/baselines_friends/artifacts` | checkpoint các baseline mở rộng (baseline study) |
| `UTS_RESULTS` | `LAYER2/output` | nơi ghi kết quả thí nghiệm RQ3–RQ5 |
| `UTS_L1_RESULTS` | `LAYER1/output` của repo | kết quả RQ1/RQ2 mà RQ3b đọc lại |

Máy 16 GB RAM: đặt `OPENBLAS_NUM_THREADS=2` và `CUDA_VISIBLE_DEVICES=` khi chạy
`rq3a_identify.py`, `02_calibrate_novelty.py` nếu gặp lỗi cấp phát bộ nhớ.
