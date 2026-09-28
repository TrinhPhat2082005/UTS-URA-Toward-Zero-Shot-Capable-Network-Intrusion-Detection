# Metric cho Layer 2 (VLM)

Tài liệu chốt các chỉ số đánh giá tầng VLM, nguồn dữ liệu, và những chỗ
paper hiện đang lệch cần sửa.

**Ràng buộc nền:** mọi chỉ số phải đến từ **một lượt chạy duy nhất**, **một
prompt duy nhất** (`neutral`). Ngoại lệ có chủ đích: E3 prompt sensitivity —
ở đó prompt chính là biến được thao tác, phải nói rõ trong paper.

---

## Quyết định nền: bỏ hướng "VLM phủ định Layer 1"

*(chốt 2026-09-27, thay thế thiết kế correction/retention trước đó)*

Thiết kế cũ đo **correction rate** — tỷ lệ VLM bác bỏ được false positive của
gate. Hướng đó đã bị loại vì ba lý do, theo thứ tự nghiêm trọng:

**1. Vòng lặp logic.** RAG truy hồi bằng **chính vector SHAP của Layer 1**. Dùng
kết quả truy hồi đó để phủ định Layer 1 là lập luận vòng tròn: bằng chứng và
thứ cần phản biện đến từ cùng một nguồn.

**2. Bằng chứng thực nghiệm phủ định giả định ban đầu.** Đo trên 5 false
positive: **cả 5 đều khớp ATTACK profile với sim = 1.000** (4/5 khớp
`Infiltration`), không cái nào khớp Benign. Lý do là logic — những flow bị báo
nhầm chính là những flow có vector SHAP *giống attack*, trong khi `label__Benign`
chỉ chứa benign mà gate phân loại **đúng**. Nghĩa là RAG không những không giúp
bác bỏ, mà còn đẩy VLM về phía **giữ** alert.

**3. Nhãn Benign của CSE-CIC-IDS2018 không đáng tin.** Audit DistriNet ghi nhận
attack thật bị gán nhãn Benign ở đúng hai ngày Infiltration; Liu et al. ước tính
sai nhãn 7.53% toàn dataset. "Ground truth" cho nhóm FP vì vậy không đủ vững để
làm mẫu số cho một chỉ số bán chính.

### Hệ quả

| | Trước | Sau |
|---|---|---|
| Câu hỏi | "VLM lọc được FP không?" | "VLM sinh được report đúng và dùng được không?" |
| Nhóm FP | 527 mẫu, correction rate | **Bỏ** |
| Nhóm TP | 140 mẫu, retention | Giữ, nhưng đổi vai trò |
| RAG | phân xử alert đúng/sai | **mô tả loại tấn công** |
| Pool SHAP | 15 nhãn (có Benign) | **14 nhãn**, không Benign |
| Profile KB | 16 file | **15 file** (bỏ `benign.json`) |

---

## Thiết kế mẫu

| Nhóm | n | Cách lấy |
|---|---|---|
| TP (alert đúng) | **140** | 10 mẫu × 14 nhãn, seed 42 |

Nguồn: Layer 1 (`ae_xgboost`, threshold 0.5828) trên `data/val_full.csv`
→ 27,266 TP. Lấy 10/nhãn vì nhãn nhỏ nhất (FTP_BruteForce) chỉ có 10 alert —
đó là mức cao nhất mà mọi nhãn đều đạt. Đủ 14/14 nhãn.

Ước tính ~35s/mẫu → **~1.4 giờ GPU**.

---

## G1 — Attack identification (xương sống mới)

Đây là câu hỏi trung tâm: **VLM có xác định đúng loại tấn công không?**

### M1. Attack type accuracy

```
acc = #(predicted_class khớp Label_Text) / #(báo cáo parse được)
```

**Đo cái gì:** VLM nói đúng **loại** tấn công, không chỉ "có tấn công".

**Vì sao quan trọng:** đây là thứ analyst thực sự dùng. Một report nói "có tấn
công" mà không nói đúng loại thì không hành động được.

**Tín hiệu từ run cũ:** `results_26_xgb.jsonl` cho **0/19**. Ví dụ RAG nói
`SSH-Bruteforce`, VLM nói *"Endpoint DoS: Service Exhaustion Flood
(Slowhttptest), T1499.002"*.

**Khớp chuỗi:** chuẩn hoá về lowercase, bỏ `_`/`-`/space, so prefix. Ca mơ hồ
đếm riêng, không tính là đúng.

### M2. Per-class accuracy

M1 tách theo từng nhãn (n=10 mỗi nhãn).

**Đo cái gì:** VLM đúng đều, hay chỉ đúng vài loại dễ (DDoS) và sai sạch các loại
khó (web attack, brute force)?

**Giới hạn phải ghi rõ:** n=10 → CI ±14–23pp. Chỉ dùng phát hiện sập hẳn
(0/10, 1/10), **không so sánh các nhãn với nhau**.

### M3. MITRE technique accuracy

```
acc_mitre = #(mitre_id khớp mitre_id của profile đúng) / #(có mitre_id)
```

**Đo cái gì:** ánh xạ sang MITRE ATT&CK có đúng không. Khác M1 ở chỗ nhiều nhãn
chia chung một technique (các biến thể DoS đều là T1499.x), nên M3 khoan dung hơn
— chênh lệch M1 vs M3 cho biết VLM sai *loại cụ thể* hay sai *cả họ tấn công*.

---

## G2 — Grounding quality

Nhóm này trả lời: **report dựa vào bằng chứng, hay tự bịa?**

### M4. RAG adherence

```
adherence = #(predicted_class khớp rag_predicted_label) / #(parse được)
```

**Đo cái gì:** VLM bám gợi ý truy hồi hay tự suy diễn.

**Run cũ:** 12/19 bám, 7/19 lệch.

### M5. Deviation harm rate

```
harm = #(lệch RAG ∧ sai ground truth) / #(lệch RAG)
```

**Đo cái gì:** chỉ số quan trọng nhất của G2.

- harm ≈ 1.0 → mỗi lần VLM "tự nghĩ" là một lần sai → **grounding là bắt buộc**
- harm thấp → VLM sửa được lỗi RAG → đóng góp ngược lại

### M6. RAG ceiling vs VLM accuracy

So M1 (VLM đúng) với **độ chính xác của chính RAG** (`rag_predicted_label` khớp
`Label_Text`).

**Đo cái gì:** VLM **cộng thêm** hay **trừ đi** so với chỉ dùng RAG? Đây là
baseline quan trọng nhất của cả thí nghiệm:

| Kết quả | Kết luận |
|---|---|
| VLM ≈ RAG | VLM chỉ chép lại — tầng này không cộng thêm gì về attribution |
| VLM < RAG | VLM **làm hỏng** kết quả truy hồi |
| VLM > RAG | VLM sửa được lỗi RAG — đóng góp thật |

### M7. Schema compliance

Tỷ lệ report đủ 6 mục theo §3.6.1, verdict đúng từ vựng, MITRE bị void khi
`FALSE_POSITIVE`. Parser đã bắt ca vi phạm (`parse_error`).

**Đo cái gì:** output có máy đọc được không — điều kiện để tích hợp vào SOC.

---

## G3 — Report quality

Nhóm này đo **chất lượng nội dung**, phần khó định lượng nhất.

### M8. Evidence grounding rate

Tỷ lệ report mà mục 3 (Forensic reasoning) **nhắc đúng tên feature** có trong
top-5 SHAP của chính flow đó.

```
grounded = #(report nhắc ≥1 feature trong top-5 SHAP) / #(parse được)
```

**Đo cái gì:** report có bám vào bằng chứng attribution thật, hay chỉ viết văn
chung chung. Đo tự động được bằng khớp chuỗi tên feature.

### M9. Hallucinated evidence rate

Tỷ lệ report nhắc đến feature/chỉ số **không có** trong input (không thuộc 95
chiều, không thuộc 5 ảnh).

**Đo cái gì:** VLM bịa bằng chứng ở mức nào. Đây là rủi ro lớn nhất khi đưa LLM
vào forensic.

**Cách đo:** trích các cụm giống tên feature trong mục 3, đối chiếu với
`feature_names`. Cần rà tay một phần để chốt tiêu chí.

### M10. Convergence statement rate

Tỷ lệ report có câu kết luận rõ ràng về hội tụ/mâu thuẫn giữa ba nguồn bằng
chứng (SHAP / biểu đồ / RAG), theo yêu cầu prompt.

**Đo cái gì:** VLM tuân thủ yêu cầu lập luận, không chỉ liệt kê.

---

## G4 — Cost

### M11. Latency per stage

mean / median / p95 cho: Layer 1 + SHAP, RAG k-NN, vẽ 5 ảnh, VLM, tổng.

**Đo cái gì:** đường nóng (gate ~20ms + RAG ~4.2ms) đủ nhanh cho real-time,
VLM (~35s) chạy bất đồng bộ nên không chặn.

---

## Bảng tổng hợp

| ID | Metric | Nhóm | n | Trả lời |
|---|---|---|---|---|
| M1 | Attack type accuracy | G1 | 140 | Đúng loại tấn công? |
| M2 | Per-class accuracy | G1 | 10/nhãn | Đúng đều hay chỉ vài loại? |
| M3 | MITRE accuracy | G1 | 140 | Ánh xạ ATT&CK đúng? |
| M4 | RAG adherence | G2 | 140 | Bám bằng chứng? |
| M5 | Deviation harm | G2 | #lệch | Tự nghĩ có hại? |
| M6 | **VLM vs RAG ceiling** | G2 | 140 | **VLM cộng thêm hay trừ đi?** |
| M7 | Schema compliance | G2 | 140 | Máy đọc được? |
| M8 | Evidence grounding | G3 | 140 | Bám SHAP thật? |
| M9 | Hallucinated evidence | G3 | 140 | Bịa bằng chứng? |
| M10 | Convergence statement | G3 | 140 | Có lập luận? |
| M11 | Latency per stage | G4 | 140 | Real-time? |

Bản ghi `parse_error` (không đọc được) **không vào mẫu số**, đếm riêng ở
`n_unparsed`. Run v8 cũ có 35/40 parse fail mà vẫn ra "tỷ lệ" — đó là số rác.

**M6 là chỉ số bán chính.** Nó là baseline duy nhất cho biết tầng VLM có đáng
tồn tại không, và không phụ thuộc kết quả tốt hay xấu.

---

## Nguồn dữ liệu

Từ `artifacts/results_<variant>_<prompt>.jsonl` do `pipeline.append_jsonl()` ghi:

| Trường | Dùng cho |
|---|---|
| `Label_Text` | M1, M2, M3, M5, M6 |
| `predicted_class` | M1, M2, M4, M5, M6 |
| `mitre_id` | M3, M7 |
| `rag_predicted_label` | M4, M5, M6 |
| `rag_is_potential_zero_day`, `rag_top_sim` | E2 zero-shot |
| `reasoning`, `raw_output` | M8, M9, M10 |
| `verdict` | M7 |
| `parse_error`, `run_error` | mọi metric (loại khỏi mẫu số) |
| `t_layer1_s`, `t_rag_s`, `t_images_s`, `t_vlm_s`, `latency_s` | M11 |
| `prompt_variant` | E3 |

**Cần bổ sung vào `pipeline.py`:** lưu `shap_top_features` (tên + giá trị top-5)
vào mỗi bản ghi, để M8/M9 tính được mà không phải chạy lại SHAP.

---

## Chỗ paper cần sửa

### 1. §4.5 — viết lại hoàn toàn

Hiện mô tả thiết kế correction/retention với "equal-sized sample from each
group". Thiết kế đó đã bị loại. Cần viết lại theo hướng report quality, kèm lý
do bỏ nhóm FP (vòng lặp logic + nhãn Benign không tin được).

### 2. §3.6.2 — cần sửa

Đoạn này lập luận về cặp correction/retention và baseline suy biến. Lập luận vẫn
đúng về mặt khái niệm nhưng **không còn là thí nghiệm được chạy**. Hai lựa chọn:

- Giữ đoạn lập luận, thêm một đoạn giải thích vì sao không đo được trên dataset này
- Rút gọn, chuyển trọng tâm sang vai trò mô tả

Tôi nghiêng về phương án 1 — nó biến một hạn chế thành phân tích có chiều sâu.

### 3. §3.5 — vai trò của RAG

Cần nói rõ reference set **chỉ chứa 14 loại tấn công, không có Benign**, và
retrieval trả lời câu hỏi "giống loại tấn công nào", không phải "alert đúng hay
sai". Kèm lý do vòng lặp logic.

### 4. §4.6 Evaluation Metrics — vẫn là comment rỗng

---

## Thứ tự thực hiện

| | Thí nghiệm | Chi phí | Nếu tốt | Nếu xấu |
|---|---|---|---|---|
| 1 | **E1 report quality** 140 TP | ~1.4h | "VLM xác định đúng X% loại tấn công" | "M6 cho thấy VLM không cộng thêm so với RAG" |
| 2 | Phân tích G2/G3 trên **chính data E1** | 0 | "Report bám bằng chứng" | "Hallucination rate X%" |
| 3 | **E3 prompt sensitivity** 3 × 20 | ~35ph | "Kết quả ổn định" | "Nhạy prompt — giới hạn paradigm" |
| 4 | **E2 zero-shot** 50 alert từ 1 fold | ~30ph | "Mô tả được cái lạ" | "Cờ zero-day vẫn hoạt động" |

Bước 1–2 bắt buộc. Bước 3–4 bonus, cả hai đều có fallback dù kết quả xấu.

**Định vị:** dựa trên run cũ (attribution 0/19, verdict flip 37/40), khả năng cao
kết quả cho thấy VLM chưa cộng thêm giá trị so với RAG. Nếu vậy, đóng góp nên
định vị là *"đo được trần chất lượng của VLM-as-forensic-reporter trên NIDS, và
chỉ ra rằng attribution vẫn đến từ tầng truy hồi chứ không từ mô hình ngôn ngữ"*
— vẫn là đóng góp Q1, và không phụ thuộc kết quả tốt hay xấu.
