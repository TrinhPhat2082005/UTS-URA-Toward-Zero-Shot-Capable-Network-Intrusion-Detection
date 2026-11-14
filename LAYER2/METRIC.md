# Đánh giá Layer 2 (VLM)

Mọi chỉ số đến từ **một lượt chạy** của `vlm_run.ipynb` trên cùng `samples.csv`, cùng prompt (`prompts.py`) và
cùng KB (sha256 ghi trong `run_meta.json`). Ablation (`KB_MODE = "off"`, bớt nhóm trong `KB_PARTS`) chạy lại
cùng mẫu, ghi vào thư mục riêng.

Đánh giá chính: **faithfulness** và **groundedness** của báo cáo (chốt 2026-11-13). Độ đúng lớp / technique và
latency là chỉ số phụ, để kiểm tra pipeline.

---

## Quyết định nền: không dùng VLM để phủ định Layer 1

*(chốt 2026-09-27)*

1. **Vòng lặp logic.** Tra cứu dùng chính vector SHAP của Layer 1; dùng kết quả đó để bác bỏ Layer 1 là lập
   luận vòng tròn.
2. **Thực nghiệm.** 5 false positive đo được đều khớp tiền lệ tấn công với khoảng cách gần 0: flow bị báo nhầm
   có SHAP giống tấn công, nên tra cứu đẩy VLM về phía giữ alert.
3. **Nhãn Benign của CSE-CIC-IDS2018 không đáng tin** (audit DistriNet; Liu et al. 2022 ước tính 7,53% sai nhãn).

Hệ quả: chỉ đánh giá alert đúng (TP); tập tham chiếu R chỉ có 14 loại tấn công, không có Benign; VLM viết báo
cáo cho alert, không phân xử alert đúng hay sai.

## Mẫu

| | |
|---|---|
| Nguồn | `val_full.csv` (Protocol B) |
| Chọn | 10 alert TP mỗi nhãn × 14 nhãn = **140**, seed 42 (`experiments/rq4_vlm/prepare_kaggle_dataset.py`) |
| Định danh | `sample_id = <Label_Text>__r<chỉ số dòng trong val_full.csv>` |
| Chạy | notebook chạy lại toàn bộ từ dòng thô: Layer 1 → detect → SHAP → tra cứu → KB → VLM |
| Nhánh | cờ tắt (có tiền lệ gần, s(q) ≤ τ_zd) và cờ bật (không có tiền lệ gần): báo cáo theo từng nhánh |

Macro trên 12 nhãn bỏ FTP-BruteForce và DoS-SlowHTTPTest (tấn công thất bại, flow trùng lặp; Liu et al. 2022);
in thêm macro 14.

## Chỉ số

### Groundedness — trích dẫn có thật và đúng chỗ (code, `metrics.py`)

Mỗi câu trong báo cáo phải kết thúc bằng id của bằng chứng hỗ trợ: `[I1]–[I5]` ảnh, `[E1]–[E7]` bằng chứng số
của alert, `[C1]..` dòng bảng C, `[K:<id>]` đoạn KB. Tập id hợp lệ của từng alert lưu ở `text/contexts.json`.

| Chỉ số | Định nghĩa | Trường |
|---|---|---|
| Citation precision | tỉ lệ id được trích có thật trong prompt của alert đó | `citation_precision`, `n_unknown_citations` |
| Citation coverage | số id trích theo từng mục báo cáo | `report.json → citations_by_section` |
| Mitigation grounding | mục 5 chỉ chọn trong ứng viên `[K:G-..]` của alert | `mitigations_valid`, `mitigations_chosen` |
| Không gán technique khi cờ bật | nhánh cờ bật: mục 4 không có T-ID | `technique_ok` (nhánh bật) |
| Schema | đủ 6 mục theo đúng tiêu đề | `n_sections` |

### Faithfulness — nội dung khớp nguồn được trích (RAGAS)

Chạy RAGAS trên `ragas.jsonl`: `user_input` (prompt), `retrieved_contexts` (khối bằng chứng, kết quả tra cứu,
bảng C và từng đoạn KB đã gửi), `response` (báo cáo). Chỉ số chính: **faithfulness** (tỉ lệ khẳng định trong
báo cáo được ngữ cảnh hỗ trợ).

Giới hạn: RAGAS chỉ đọc text, nên khẳng định rút ra từ ảnh (`[I1]–[I5]`) không kiểm được bằng RAGAS; số liệu
của ảnh SHAP đã có dạng text trong khối bằng chứng (`[E1]–[E5]`).

### Phụ — đúng lớp, đúng technique (code)

| Chỉ số | Định nghĩa | Trường |
|---|---|---|
| Lookup accuracy | nhãn vote 5-NN = nhãn thật | `pred_label`, `label_true` |
| Class | cờ tắt: `Predicted class` = nhãn thật; cờ bật: `UNKNOWN` | `class_correct` |
| Technique | cờ tắt: technique ở mục 4 thuộc technique của nhãn theo Borisenko 2022 | `technique_ok` |
| Đánh giá nhánh bật | variant / different / undetermined | `assessment` |

### Latency từng chặng

`time.perf_counter`, một flow mỗi lần, sau warm-up; `latency.csv` (giây), `metrics.latency_table` (ms: mean,
median, p95). Chặng: `layer1` · `shap` · `lookup` · `contrast` · `kb_select` · `images` · `prompt` · `vlm` ·
`parse`. Đo riêng phần phát hiện (20 warm-up + 300 lần, CPU): `experiments/rq5_latency/latency_detection_path.py`.

## Nguồn dữ liệu

| File | Nội dung |
|---|---|
| `results.jsonl` | một dòng / sample: nhãn, p(x), tra cứu, cờ, technique, kết quả parse, các chỉ số code, latency |
| `<sample_id>/output/report.json` | báo cáo đã parse: mục, trích dẫn theo mục, mitigation đã chọn, id không hợp lệ |
| `<sample_id>/text/` | prompt đầy đủ + tập ngữ cảnh / id hợp lệ |
| `ragas.jsonl` | đầu vào RAGAS |
| `latency.csv` | latency từng chặng |
| `run_meta.json` | commit, sha256 KB, τ_α, τ_zd, cấu hình VLM, KB_MODE / KB_PARTS |

## Chỗ paper cần khớp

- §3.5: tra cứu trả lời "giống loại tấn công đã biết nào" (vote 5-NN trên SHAP, R không có Benign), cờ khi
  s(q) > τ_zd; không phân xử alert.
- §3.6: KB khóa nhãn → technique theo Borisenko et al. 2022 Table II; nhánh cờ bật dùng `nearest_background`.
- §4.5–4.6: mẫu 140 TP, hai nhánh; faithfulness (RAGAS) + groundedness (trích dẫn) là chỉ số chính.
