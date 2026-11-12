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
| `gate/engine.py` | 3.3 | `Layer1Engine`: nạp checkpoint Layer 1 đã train, Protocol B (`Layer1Engine(variant)`) hoặc một fold Protocol A (`Layer1Engine.from_fold`), tính p(x) và vector SHAP. **Nơi duy nhất tính SHAP** (`shap.TreeExplainer`). LAYER2 không train Layer 1: kiến trúc AE/SVDD import từ `LAYER1/03_training/models` (`config.LAYER1_CODE_DIR`, biến `UTS_LAYER1_CODE`) |
| `retrieval/reference_set.py` | 3.5 | Tập tham chiếu R: đường dẫn, hàm ghi, build Protocol B. Mỗi loại lưu φ (`label__`), giá trị thô 78 cột (`x__`) và chỉ số dòng trong train_full (`row__`), cùng thứ tự dòng; `--add-raw` thêm `x__`/`row__` vào R đã có mà không tính lại SHAP |
| `retrieval/store.py` | 3.5 | `ReferenceStore`: nạp R một lần; `lookup(q)` → nhãn vote 5-NN (hòa → flow gần nhất), các tiền lệ cùng nhãn thắng, s(q) trên 10-NN, cờ. Thiết kế đã chốt 2026-11-13 |
| `retrieval/retriever.py` | 3.5, Eq. `type-dist`, `novelty` | `SHAPKNNRetriever` (thiết kế cũ, giữ để tái lập Setup 1/2 đã chạy): một lần tính L2 → 3 loại gần nhất + điểm novelty |
| `retrieval/novelty.py` | 3.5, Eq. `zeroday-flag` | Cờ zero-day `s(q) > τ_zd`, đọc và hiệu chỉnh τ_zd |
| `evidence/build.py` | 3.7 | Chặng A: `EvidenceBuilder` (retrieve + context + 5 ảnh), hàm ghi ứng viên và top feature cho manifest |
| `evidence/bundle.py` | | Ghi và đọc bundle: `manifest.jsonl`, `phi.npy`, `bundle_meta.json`, `images/` |
| `evidence/contrast.py` | 3.6 | Bảng đối chiếu C cho alert bị bật cờ: φ của alert so với trung bình các tiền lệ cùng nhãn thắng, kèm giá trị thô |
| `vlm/runner.py` | 3.7 | Chặng B: `VLMRunner` (Qwen2.5-VL 7B, confidence từ logprob) |
| `prompts.py` | 3.7, Fig. `vlmio` | Mọi text gửi VLM và parser: prompt báo cáo 6 mục (demo live), prompt chọn ứng viên (Setup 1/2), context của retriever (`format_retrieval_context`) |
| `utils/plot_shap.py`, `utils/plot_flow.py` | 3.7 | SHAP waterfall và 4 biểu đồ luồng |
| `utils/io.py` | | `append_jsonl`, commit của code |
| `knowledge_base/` | 3.6 | Knowledge base có nguồn (153 đoạn, mọi trích dẫn được kiểm nguyên văn): xem `knowledge_base/README.md`. `attack_profiles/` + `profiles.py` là profile cũ, giữ để tái lập Setup 1/2 đã chạy |
| `tests/` | | Test của `retrieval/store.py` và `evidence/contrast.py` (gồm kiểm khớp dòng φ ↔ giá trị thô trên R thật) |
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
| `layer1/<variant>/` | Checkpoint Layer 1 Protocol B do `LAYER1/03_training/train_protocol_b.py` ghi ra (`model.pkl`, `scaler.pkl`, `feature_names.json`, `metrics.json`, encoder). `metrics.json`: `threshold` = ngưỡng ngân sách FPR α = 0.5% trên benign train; `thresholds_by_alpha` giữ ngưỡng mọi α khác, kể cả 0.4% của các lần chạy cũ |
| `reference_sets/protocol_b/<variant>.npz` (+ `.meta.json`) | Tập R của pipeline (mọi flow attack của train_full); τ_zd ở khóa `novelty` của meta; từ 2026-11-13 có `x__`/`row__` (bản chỉ có φ: `*.phi_only.bak.npz`) |
| `kb_sources/<nguồn>/` | File nguồn của knowledge base (ATT&CK, CAPEC, RFC, IANA, mã CICFlowMeter, trang UNB...), tải bằng `knowledge_base/build/fetch_sources.py`, khóa sha256 trong `knowledge_base/store/sources.lock.json` |
| `reference_sets/protocol_a/<fold>/<variant>.npz` (+ `.meta.json`) | Tập R của từng fold LOAO (R3, Setup 2); meta có thêm τ_α của fold |

## Chạy

1. `python LAYER1/03_training/train_protocol_b.py` → `artifacts/layer1/<variant>/`
   (`--thresholds-only`: chỉ tính lại ngưỡng từ checkpoint đã có, không train lại)
2. `python LAYER2/retrieval/reference_set.py` → `artifacts/reference_sets/protocol_b/<variant>.npz`
3. `python LAYER2/experiments/rq3_retrieval/02_calibrate_novelty.py` → τ_zd vào file meta
4. Báo cáo VLM: Setup 1/2 theo `experiments/README.md`; đợt pilot cũ dùng `run_layer2.ipynb` (GPU).

Demo live (bắt gói tin thật) dùng chính các thành phần này qua `project_root/main.py`. Demo vẫn dùng prompt
báo cáo 6 mục; prompt chọn ứng viên mới dùng ở Setup 1/2.
