# LAYER2 — tra cứu trên vector SHAP, knowledge base và báo cáo VLM

Code của Section 3.4–3.7 trong paper. Thí nghiệm của paper nằm trong `experiments/` — xem
`experiments/README.md` để tra từ bảng trong paper sang script.

```
flow thô ─► Layer 1: p(x) ≥ τ_α ? ─► TreeSHAP φ ─► tra tập R: vote 5-NN (ℓ*), s(q), cờ ─► [bảng C nếu cờ bật]
            gate/engine.py                          retrieval/store.py                      evidence/contrast.py
         ─► chọn đoạn KB ─► 5 ảnh ─────────────────────────────────────────────────────────  evidence/build.py
            knowledge_base/retrieve.py   utils/plot_*.py
         ─► prompt (trích dẫn [I] [E] [C] [K:..]) ─► Qwen2.5-VL ─► báo cáo 6 mục ─► parse + chấm trích dẫn
            prompts.py                                 vlm/runner.py                   prompts.py, metrics.py
```

Toàn bộ chuỗi cho **một sample** nằm ở `pipeline.run_sample`; notebook duy nhất `vlm_run.ipynb` (sinh bởi
`make_vlm_notebook.py`) chạy nó cho từng dòng của `samples.csv` trên Kaggle, đo latency từng chặng và ghi:

```
<RUN_DIR>/<sample_id>/evidence/   5 ảnh, top5.json, evidence.json (tra cứu, bảng C, đoạn KB)
<RUN_DIR>/<sample_id>/text/       system_prompt.txt, user_prompt.txt (prompt đầy đủ trước khi gửi VLM), contexts.json
<RUN_DIR>/<sample_id>/output/     report.md, report.json
<RUN_DIR>/results.jsonl · ragas.jsonl · latency.csv · run_meta.json
```

## Cấu trúc

| Thư mục / file | Paper | Việc |
|---|---|---|
| `config.py` | | Đường dẫn và tham số; dữ liệu đọc theo biến `UTS_DATA`; `KB_MODE`, `KB_PARTS` cho ablation |
| `gate/engine.py` | 3.3 | `Layer1Engine`: nạp checkpoint Layer 1 (Protocol B hoặc một fold Protocol A), tính p(x) và φ. **Nơi duy nhất tính SHAP** (`shap.TreeExplainer`). Kiến trúc AE/SVDD import từ `LAYER1/03_training/models` (`UTS_LAYER1_CODE`) |
| `retrieval/reference_set.py` | 3.5 | Tập tham chiếu R: mỗi loại lưu φ (`label__`), giá trị thô 78 cột (`x__`), chỉ số dòng (`row__`) |
| `retrieval/store.py` | 3.5 | `ReferenceStore.lookup(q)`: nhãn vote 5-NN (hòa → flow gần nhất), các tiền lệ thắng, s(q) trên 10-NN, cờ |
| `retrieval/novelty.py` | 3.5, Eq. `zeroday-flag` | Cờ `s(q) > τ_zd`; đọc và hiệu chỉnh τ_zd |
| `evidence/build.py` | 3.7 | Chặng A cho một flow: Layer 1 → detect → SHAP → tra cứu → bảng C → KB → 5 ảnh, đo thời gian từng chặng |
| `evidence/contrast.py` | 3.6 | Bảng đối chiếu C (alert bị bật cờ so với trung bình tiền lệ cùng nhãn, φ và x) |
| `knowledge_base/` | 3.6 | KB có nguồn (khóa nhãn → technique theo Borisenko 2022 Table II): xem `knowledge_base/README.md`, bản đồ `knowledge_base/kb_map.html` |
| `prompts.py` | 3.7, Fig. `vlmio` | Prompt hai nhánh (cờ tắt / bật, khối `nearest_background`), schema 6 mục, parser + trích dẫn |
| `vlm/runner.py` | 3.7 | `VLMRunner` (Qwen2.5-VL 7B, 4-bit, confidence từ logprob của `Predicted class`) |
| `pipeline.py` | | `run_sample` (toàn bộ chuỗi cho một dòng + ghi kết quả), `warmup`, `write_run_meta`, `done_ids` |
| `metrics.py` | 4.5 | Tổng hợp một lần chạy: đúng lớp / technique, độ chính xác trích dẫn, mitigation hợp lệ, latency từng chặng |
| `make_vlm_notebook.py` → `vlm_run.ipynb` | 4.5 | Notebook duy nhất chạy VLM (Kaggle hoặc local) |
| `utils/plot_shap.py`, `utils/plot_flow.py` | 3.7 | SHAP waterfall và 4 biểu đồ luồng (biểu đồ cờ dán nhãn theo cờ cột **thực chứa**) |
| `utils/timing.py`, `utils/io.py` | | `StageTimer` (perf_counter từng chặng), `append_jsonl`, commit của code |
| `tests/` | | Test của `retrieval/store.py` và `evidence/contrast.py` |

`METRIC.md` là ghi chép quyết định đánh giá của đợt pilot cũ (bỏ hướng "VLM phủ định Layer 1"); chỉ số hiện tại
nằm ở `metrics.py` và RAGAS trên `ragas.jsonl`.

## Artifact (`UTS_DATA/artifacts/`)

| Đường dẫn | Nội dung |
|---|---|
| `layer1/<variant>/` | Checkpoint Layer 1 Protocol B (`LAYER1/03_training/train_protocol_b.py`): `model.pkl`, `scaler.pkl`, encoder, `feature_names.json`, `metrics.json` (`threshold` = ngân sách FPR α = 0.5% trên benign train) |
| `reference_sets/protocol_b/<variant>.npz` (+ `.meta.json`) | Tập R của pipeline (mọi flow attack của train_full, có `x__`/`row__`); τ_zd ở khóa `novelty` của meta |
| `reference_sets/protocol_a/<fold>/<variant>.npz` (+ `.meta.json`) | Tập R từng fold LOAO; meta có thêm τ_α của fold |

File nguồn của KB nằm ở `UTS_DATA/kb_sources/raw/<id>/` (xem `knowledge_base/README.md`).

## Chạy

1. `python LAYER1/03_training/train_protocol_b.py` → `artifacts/layer1/<variant>/`
2. `python LAYER2/retrieval/reference_set.py` → `artifacts/reference_sets/protocol_b/<variant>.npz`
3. `python LAYER2/experiments/rq3_retrieval/02_calibrate_novelty.py` → τ_zd vào file meta
4. `python LAYER2/experiments/rq4_vlm/prepare_kaggle_dataset.py` → `UTS_DATA/kaggle_upload/layer2-vlm-run/`
   (samples.csv: 10 alert TP / nhãn + checkpoint + tập R); upload thành Kaggle dataset
5. `python LAYER2/make_vlm_notebook.py` → `vlm_run.ipynb`; chạy trên Kaggle (GPU) theo hướng dẫn trong notebook

Demo live (`project_root/main.py`) vẫn dùng retriever cũ và chưa được chuyển sang chuỗi mới.
