# LAYER2 — VLM forensic report

Đánh giá tầng VLM như một **second opinion** trên các alert Layer 1 đã raise,
theo Methodology §3.6 và Experimental Setup §4.5.

## Chạy

Mở `run_layer2.ipynb`, chạy tuần tự từ trên xuống. Cell 8 chạy **một sample ra
một report** — xem kỹ output đó trước khi chạy cả lô ở cell 9.

```
row → Layer 1 (proba + SHAP) → RAG k-NN → 5 ảnh → VLM → report
```

Mỗi sample đi hết chuỗi ngay trong lúc chạy. Không build evidence trước như
`experiments/vlm_gate/prepare_evidence_*.py`, nên đổi mẫu hoặc đổi prompt không
phải dựng lại ảnh.

## File

| File | Việc |
|---|---|
| `config.py` | Toàn bộ đường dẫn + tham số. Sửa ở đây, không sửa trong notebook. |
| `prompts.py` | System prompt, schema báo cáo, parser, 3 prompt variant |
| `pipeline.py` | `Layer1Engine`, `EvidenceBuilder`, `VLMRunner`, `analyze_sample` |
| `metrics.py` | correction/retention + Wilson CI, quét confidence, latency |
| `shap_knn_retriever.py` | RAG: cosine k-NN trên vector SHAP (copy từ `experiments/`) |
| `flow_visualizer.py` | 4 biểu đồ luồng (copy từ `experiments/`) |
| `run_layer2.ipynb` | Notebook chính |
| `_make_notebook.py` | Sinh lại notebook nếu cần |

## Hai quyết định thiết kế, và lý do

### 1. Confidence lấy từ logprob, không dùng số model tự khai

Đo lại các run cũ (`experiments/vlm_gate/artifacts/`): v6/v7/v13 chạy trên
**đúng cùng 40 mẫu**, confidence tự khai gần như luôn = 0.9 — v7 khai 0.9 cho
cả 40/40 mẫu kể cả khi sai toàn bộ nhóm FP.

Một cột hằng số thì không quét ngưỡng được. Mà §4.5 đã hứa *"the effect of a
confidence threshold on either rate can be read off"*. Nên `VLMRunner` lấy
softmax prob của token verdict (`confidence_logprob`), và vẫn giữ số tự khai ở
`confidence_stated` để đối chiếu. Cell 11 so hai cột.

### 2. Prompt trung tính + giữ lại các biến thể lệch để đo

Cùng 40 mẫu đó, **37/40 đổi verdict chỉ vì đổi chữ trong prompt**:

| Run | Correction | Retention | |
|---|---|---|---|
| v7 (nhấn "Layer 1 đã flag") | 0% | 100% | nói ATTACK 40/40 |
| v13 (nhấn "hãy hoài nghi") | 90% | 20% | nói FALSE_POSITIVE 34/40 |

Hai cực suy biến mà §3.6.2 gọi là *"trivial to maximize alone"*. Chọn version
có số đẹp nhất là chọn theo nhiễu prompt, không phòng thủ được trước reviewer.

Nên: `neutral` là mặc định, `trust_l1` và `skeptical` giữ lại trong
`PROMPT_VARIANTS` để **đo độ nhạy** (cell 14), không phải để chọn cái tốt nhất.
Nếu tỷ lệ flip vẫn cao thì đó là kết quả phải báo cáo — nó chứng minh đúng lập
luận §3.6.2 rằng tầng này chưa nên tự động hoá.

## Ghi chú

- **Verdict vocabulary khoá cứng** ở `CONFIRMED_ATTACK` / `FALSE_POSITIVE`.
  Run cũ lẫn lộn với `ATTACK`/`BENIGN` nên không gộp bảng được.
- **Bản ghi parse lỗi không vào mẫu số** của correction/retention, đếm riêng ở
  `n_unparsed`. Run v8 cũ có 35/40 parse fail mà vẫn ra được "tỷ lệ" — đó là số rác.
- **Nhóm FP hiếm**: ở ngưỡng 0.7 trên 40K dòng val_binary chỉ có 37 FP so với
  7,956 TP. Đây chính là lý do §4.5 phải cân bằng hai nhóm. Nếu `sample_balanced`
  báo lấy được ít hơn N thì là do hết FP, không phải lỗi.
- `Timestamp` nằm giữa các cột số nhưng **không phải feature** — đã đưa vào
  `META_COLS`. Bỏ nó đi thì 78 cột còn lại khớp đúng thứ tự với `feature_names`
  trong `shap_full_*.npz`.
- Chạy lại cell 9 sẽ **ghi tiếp** vào JSONL cũ và bỏ qua sample đã xong; xoá file
  nếu muốn làm lại từ đầu.

## Đã kiểm chứng

Chạy thật trên máy local (trừ VLM — cần GPU):

- `config.describe()` — 7/7 đường dẫn tồn tại
- Layer 1 + TreeExplainer — SHAP shape `(1, 78)`
- RAG k-NN — 14 label pool, khớp `DoS attacks-Hulk` sim=1.00, **4.2 ms**
- 5 ảnh — render đủ, 45–96 KB mỗi ảnh, **1.4 s**
- Parser — đúng trên cả 4 ca: hợp lệ / FALSE_POSITIVE kèm MITRE (bắt được vi
  phạm schema) / rỗng / rác
- Metrics — tái tạo đúng tỷ lệ đã bơm vào, CI Wilson, loại bản ghi lỗi khỏi mẫu số
