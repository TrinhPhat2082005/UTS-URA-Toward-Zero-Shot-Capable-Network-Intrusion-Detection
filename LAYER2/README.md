# LAYER2 — SHAP-vector retrieval và báo cáo VLM

Code của Section 3.4–3.7 trong paper. Thí nghiệm của paper nằm trong `experiments/` — xem
`experiments/README.md` để tra từ bảng trong paper sang script.

```
flow ─► Layer 1: p(x) + TreeSHAP ─► retriever: top-3 + cờ zero-day ─► context + 5 ảnh   ┐ chặng A (CPU)
        layer1/engine.py             retrieval/                        evidence/build.py ┘
     ─► prompt ─► Qwen2.5-VL ─► báo cáo                                                   ┐ chặng B (GPU)
        prompts.py  vlm/runner.py                                                         ┘
```

Hai chặng nối với nhau bằng **evidence** của từng alert. Evidence gồm: xác suất, φ, 3 ứng viên kèm khoảng
cách, novelty, cờ zero-day và 5 ảnh.
- **Demo live** (`project_root/main.py`, `pipeline.analyze_sample`) chạy hai chặng liền nhau.
- **Setup 1/2** ghi chặng A thành bundle (`evidence/bundle.py`), upload lên Kaggle, rồi chạy chặng B ở đó.
  Chặng B chỉ cần bundle, `prompts.py`, `knowledge_base/` và `vlm/`: không cần Layer 1 hay tập R.

## Cấu trúc

| Thư mục / file | Paper | Việc |
|---|---|---|
| `config.py` | | Đường dẫn và tham số; dữ liệu đọc theo biến `UTS_DATA` |
| `layer1/engine.py` | 3.3 | `Layer1Engine`: nạp checkpoint Protocol B (`Layer1Engine(variant)`) hoặc một fold Protocol A (`Layer1Engine.from_fold`), tính p(x) và vector SHAP. **Nơi duy nhất tính SHAP** (`shap.TreeExplainer`) |
| `layer1/train.py` | 3.3, Eq. `tau-alpha` | Train Layer 1 Protocol B (config `xgb_config_mono.py` của LAYER1), ngưỡng theo ngân sách FPR α = 0.4% |
| `layer1/models/` | | Kiến trúc AE/SVDD (giống hệt `LAYER1/03_training/models`), để LAYER2 tự chạy được trên Kaggle |
| `retrieval/reference_set.py` | 3.5 | Tập tham chiếu R: đường dẫn, hàm ghi, build Protocol B |
| `retrieval/retriever.py` | 3.5, Eq. `type-dist`, `novelty` | `SHAPKNNRetriever`: một lần tính L2 → 3 loại gần nhất + điểm novelty. Chỉ trả số và profile, không viết text |
| `retrieval/novelty.py` | 3.5, Eq. `zeroday-flag` | Cờ zero-day `s(q) > τ_zd`, đọc và hiệu chỉnh τ_zd |
| `evidence/build.py` | 3.7 | Chặng A: `EvidenceBuilder` (retrieve + context + 5 ảnh), hàm ghi ứng viên và top feature cho manifest |
| `evidence/bundle.py` | | Ghi và đọc bundle: `manifest.jsonl`, `phi.npy`, `bundle_meta.json`, `images/` |
| `vlm/runner.py` | 3.7 | Chặng B: `VLMRunner` (Qwen2.5-VL 7B, confidence từ logprob) |
| `prompts.py` | 3.7, Fig. `vlmio` | Mọi text gửi VLM và parser: prompt báo cáo 6 mục (demo live), prompt chọn ứng viên (Setup 1/2), context của retriever (`format_retrieval_context`) |
| `utils/plot_shap.py`, `utils/plot_flow.py` | 3.7 | SHAP waterfall và 4 biểu đồ luồng |
| `utils/io.py` | | `append_jsonl`, commit của code |
| `knowledge_base/` | 3.6 | 14 profile MITRE ATT&CK + unknown_anomaly (`attack_profiles/`), bảng nhãn → profile (`profiles.py`) |
| `pipeline.py` | | `analyze_sample` (hai chặng cho một dòng), lấy mẫu alert, vẽ lại ảnh; vẫn gọi được `pipeline.Layer1Engine/EvidenceBuilder/VLMRunner/append_jsonl` như trước |
| `metrics.py`, `METRIC.md`, `run_layer2.ipynb`, `_make_notebook.py` | 4.5 | Đợt pilot với prompt báo cáo 6 mục (giữ nguyên) |

## Bật / tắt UNKNOWN

Có hai việc khác nhau:
- **Cờ zero-day** (`retrieval/novelty.py`): `s(q) > τ_zd`. Cờ luôn được tính và ghi vào evidence.
- **Có đưa lựa chọn UNKNOWN cho VLM hay không**: tham số `allow_unknown` của `prompts.build_candidate_prompt`.
  - Setup 1 đặt `False`: mọi alert đều thuộc loại đã biết.
  - Setup 2 và pipeline đặt `True`: UNKNOWN xuất hiện khi cờ bật.

Vì công tắc nằm ở prompt, cùng một bundle chạy được cả hai chế độ mà không cần build lại evidence.

## Artifact (`UTS_DATA/artifacts/`)

| Đường dẫn | Nội dung |
|---|---|
| `layer1/<variant>/` | Checkpoint Layer 1 Protocol B (`model.pkl`, `scaler.pkl`, `feature_names.json`, `metrics.json`, encoder) |
| `reference_sets/protocol_b/<variant>.npz` (+ `.meta.json`) | Tập R của pipeline (mọi flow attack của train_full); τ_zd ở khóa `novelty` của meta |
| `reference_sets/protocol_a/<fold>/<variant>.npz` (+ `.meta.json`) | Tập R của từng fold LOAO (R3, Setup 2); meta có thêm τ_α của fold |

## Chạy

1. `python LAYER2/layer1/train.py` → `artifacts/layer1/<variant>/`
2. `python LAYER2/retrieval/reference_set.py` → `artifacts/reference_sets/protocol_b/<variant>.npz`
3. `python LAYER2/experiments/rq3_retrieval/02_calibrate_novelty.py` → τ_zd vào file meta
4. Báo cáo VLM: Setup 1/2 theo `experiments/README.md`; đợt pilot cũ dùng `run_layer2.ipynb` (GPU).

Demo live (bắt gói tin thật) dùng chính các thành phần này qua `project_root/main.py`. Demo vẫn dùng prompt
báo cáo 6 mục; prompt chọn ứng viên mới dùng ở Setup 1/2.
