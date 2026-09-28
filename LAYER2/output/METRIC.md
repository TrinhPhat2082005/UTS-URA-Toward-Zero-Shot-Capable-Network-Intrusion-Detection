# Kết quả — Layer 2 VLM report quality

**Nguồn:** `output/results_ae_xgboost_neutral_v1.jsonl` (140 bản ghi, 10 mẫu × 14 nhãn, seed 42)
**Variant Layer 1:** `ae_xgboost` · **Prompt:** `neutral` · **Model:** Qwen2.5-VL 7B, 4-bit, greedy decoding
**Ngày chạy:** 2026-09-28 (Kaggle GPU) · **run_error:** 0/140 · **parse_error:** 0/140

Định nghĩa từng chỉ số: xem `LAYER2/METRIC.md`. File này chỉ ghi **số đo được**.

---

## 0. Sửa lỗi đo lường trước khi báo cáo

Hàm so khớp tên lớp (`metrics._match`) ban đầu chỉ so prefix sau khi bỏ `_`/`-`/khoảng
trắng, không bắt được lệch kiểu **số ít/nhiều giữa chuỗi**
(`DDoS_attacks_LOIC_HTTP` vs `DDOS_attack_LOIC_HTTP` — VLM đổi `attacks`→`attack`
và `DDoS`→`DDOS`, nội dung vẫn đúng). Điều này làm 6/140 bản ghi bị tính sai,
kéo M1 xuống 61% thay vì đúng là 65%.

Đã vá: `_match()` thêm bước so theo từng từ con, bỏ hậu tố `s`. Xác nhận không
gây nhầm chéo trên toàn bộ 14×13=182 cặp nhãn khác nhau (0 false positive).
Số trong file này là số **sau khi vá**.

---

## 1. Tổng quan (summary)

| Chỉ số | Giá trị | CI 95% / ghi chú |
|---|---|---|
| **M1** Attack type accuracy | **65.0%** (91/140) | Wilson [56.8%, 72.4%] |
| **M3** Có MITRE id | 99.3% (139/140) | |
| **M4** RAG adherence (VLM lặp lại gợi ý RAG) | 77.1% | |
| **M5** Deviation harm (trong số đi lệch RAG, tỷ lệ sai) | **100.0%** | 32 ca đi lệch, 0 ca sửa đúng |
| **M6** RAG accuracy | 80.7% (113/140) | baseline |
| **M6** VLM accuracy | 65.0% (91/140) | |
| **M6** Delta (VLM − RAG) | **−15.7pp** | **VLM làm hỏng** |
| **M7** Schema đủ 6 mục | 100.0% | |
| **M8** Evidence grounding (nhắc đúng feature SHAP) | 100.0% | |
| **M9** Hallucinated evidence (ước lượng) | 21.4% | cần rà tay xác nhận |
| **M10** Có câu kết luận hội tụ/mâu thuẫn | 100.0% | |
| Report lẫn chữ Hán | 7.1% (10/140) | trước khi ép tiếng Anh trong prompt (đợt chạy này ở giai đoạn đã sửa) |
| AUROC(confidence_logprob) | 0.467 | **không phân biệt được đúng/sai** — tệ hơn ngẫu nhiên |

### McNemar's test (RAG vs VLM, so cặp trên cùng 140 mẫu)

|  | VLM đúng | VLM sai |
|---|---|---|
| **RAG đúng** | 91 | **22 (b)** |
| **RAG sai** | **0 (c)** | 27 |

- b (RAG đúng, VLM làm sai) = 22
- c (RAG sai, VLM sửa đúng) = 0
- χ² (continuity-corrected) = 20.05, **p = 0.000008**

**Không có ca nào VLM sửa đúng một lỗi của RAG.** Toàn bộ 22 ca lệch giữa hai
mô hình đều theo hướng VLM phá một câu trả lời RAG đã đúng. Đây là bằng chứng
thống kê mạnh nhất trong bộ số này.

### Confidence sweep — hai nguồn cho kết luận trái ngược nhau

| Ngưỡng logprob | n giữ | accuracy | | Ngưỡng tự khai | n giữ | accuracy |
|---|---|---|---|---|---|---|
| 0.23 | 140 | 65.0% | | 70 | 140 | 65.0% |
| 0.39 | 130 | 64.6% | | 76 | 134 | 67.9% |
| 0.54 | 124 | 62.9% | | 82 | 130 | 69.2% |
| 0.69 | 109 | 60.6% | | 88 | 112 | 69.6% |
| 0.84 | 84 | 58.3% | | 94 | 69 | 71.0% |
| 0.99 | 1 | 100.0%* | | 100 | 22 | 72.7% |

\* n=1, không có ý nghĩa thống kê.

**Logprob nội tại: accuracy GIẢM khi ngưỡng tăng** (65.0% → 58.3% ở phần thân
đường cong, bỏ điểm cuối n=1). Model tự tin nhất (theo phân phối xác suất nội
tại) lại chính là lúc nó sai nhiều nhất — khớp với việc AUROC = 0.467 < 0.5.

**Confidence tự khai: accuracy TĂNG khi ngưỡng tăng** — đúng hướng kỳ vọng,
nhưng đây là con số model tự viết ra, không phải trạng thái nội tại đo được.

Hai chiều ngược nhau này tự nó là một phát hiện: **confidence tự khai có vẻ
"hoạt động" chỉ vì nó tương quan với việc model paraphrase gợi ý RAG một cách
tự tin, không phải vì nó phản ánh độ chính xác thật.**

---

## 2. Per-label

| Nhãn | n | RAG đúng | VLM đúng | RAG% | VLM% | Δ | |
|---|---|---|---|---|---|---|---|
| Bot | 10 | 10 | 10 | 100% | 100% | 0% | |
| DDOS_attack_HOIC | 10 | 10 | 10 | 100% | 100% | 0% | |
| DDoS_attacks_LOIC_HTTP | 10 | 10 | 10 | 100% | 100% | 0% | |
| DoS_attacks_GoldenEye | 10 | 10 | 10 | 100% | 100% | 0% | |
| SSH_Bruteforce | 10 | 10 | 10 | 100% | 100% | 0% | |
| DoS_attacks_Hulk | 10 | 9 | 9 | 90% | 90% | 0% | |
| DoS_attacks_SlowHTTPTest | 10 | 1 | 1 | 10% | 10% | 0% | RAG kém sẵn |
| SQL_Injection | 10 | 3 | 2 | 30% | 20% | −10% | RAG kém sẵn |
| Brute_Force_Web | 10 | 10 | 8 | 100% | 80% | **−20%** | VLM làm hỏng |
| DDOS_attack_LOIC_UDP | 10 | 10 | 8 | 100% | 80% | **−20%** | VLM làm hỏng |
| Brute_Force_XSS | 10 | 10 | 7 | 100% | 70% | **−30%** | VLM làm hỏng |
| DoS_attacks_Slowloris | 10 | 10 | 4 | 100% | 40% | **−60%** | VLM làm hỏng |
| Infiltration | 10 | 10 | 2 | 100% | 20% | **−80%** | VLM làm hỏng nặng nhất |
| **FTP_BruteForce** | 10 | 0 | 0 | 0% | 0% | 0% | **RAG hỏng hoàn toàn** |
| **TỔNG** | **140** | **113** | **91** | **80.7%** | **65.0%** | **−15.7%** | |

CI mỗi hàng: n=10/nhãn → Wilson CI rộng (±14–23pp tại p gần 0.5–0.9, xem
`LAYER2/METRIC.md` §M2). Bảng per-label dùng để **phát hiện loại lỗi**, không
dùng để so sánh nhãn với nhau theo nghĩa thống kê chặt.

### Ba nhóm hành vi rõ rệt

**Nhóm A — cả RAG và VLM đều hoàn hảo (5 nhãn):** Bot, HOIC, LOIC_HTTP,
GoldenEye, SSH. Traffic đặc trưng mạnh, vector SHAP tách biệt rõ.

**Nhóm B — RAG đã yếu sẵn, VLM không cứu được (3 nhãn):** SlowHTTPTest (10%),
SQL_Injection (30%), FTP_BruteForce (**0%**). Đây là vấn đề retrieval: pool
train quá mỏng (43–67 dòng cho FTP_BruteForce/SQL_Injection). Đổi prompt không
tác động vì đầu vào RAG đã sai từ trước khi tới VLM.

**Nhóm C — RAG đúng 100% nhưng VLM tự đổi thành sai (5 nhãn):** Brute_Force_Web,
DDOS_attack_LOIC_UDP, Brute_Force_XSS, Slowloris, **Infiltration**. Đây là
nhóm quyết định câu chuyện M6: RAG context trong prompt nói đúng, report
cuối cùng viết khác. Infiltration mất 80 điểm phần trăm dù RAG đưa gợi ý
đúng tuyệt đối ở cả 10/10 mẫu.

---

## 3. Hai ca lỗi định tính đáng chú ý

### 3.1. `DDOS_attack_LOIC_HTTP` — nhầm khuôn tên, không phải hallucination

VLM viết `DDOS_attack_LOIC_HTTP` (6/12 lần cho nhãn `DDoS_attacks_LOIC_HTTP`)
thay vì đúng chuỗi trong prompt. Nghi vấn: 3 tên LOIC/HOIC gần giống nhau
trong danh sách 14 lớp —

```
DDOS_attack_HOIC          <- DDOS, attack (so it)
DDOS_attack_LOIC_UDP      <- DDOS, attack (so it)
DDoS_attacks_LOIC_HTTP    <- DDoS, attacks (so nhieu)   <- nhan bi loi
```

VLM áp khuôn viết của 2 tên đầu lên tên thứ 3. Đã tính đúng sau khi vá
`_match()`; **không cần sửa VLM**, nhưng đáng cân nhắc làm nổi bật khác biệt
3 tên này trong prompt để giảm rủi ro tái diễn ở lần chạy khác.

### 3.2. `Cross-Site Scripting (T1189)` — vi phạm taxonomy, MITRE ID vẫn đúng

Một mẫu `SQL_Injection` có RAG gợi ý sai (`Brute_Force_XSS`, sim = 0.9999998).
VLM viết `predicted_class = "Cross-Site Scripting (T1189)"` — vi phạm ràng buộc
"exactly one of the 14 names" — nhưng `mitre_id = T1189` khớp **chính xác**
với profile `web_attack_xss.json`. VLM không bịa MITRE ID; nó tra đúng nguồn
cho một lớp mà RAG đã gợi ý sai. Đây là lỗi lan truyền từ RAG + vi phạm
schema, không phải hallucination theo nghĩa "bịa thông tin không tồn tại".

**Hệ quả cho thiết kế metric:** M9 hiện gộp chung "hallucinated evidence" —
nên tách `schema_violation` (dùng tên ngoài 14 nhãn) khỏi `hallucinated_feature`
(nhắc feature không có thật) ở lần đo tiếp theo.

---

## 4. Latency (M11)

| Chặng | mean | p95 |
|---|---|---|
| Layer 1 + SHAP | 0.050s | 0.06s |
| RAG k-NN | 0.003s | 0.00s |
| Vẽ 5 ảnh | 1.099s | 1.48s |
| VLM | 64.912s | 72.32s |
| **Tổng** | **66.065s** | **73.43s** |

Đường nóng (Layer 1 + RAG) ~53ms — đủ nhanh cho real-time. VLM chiếm 98% tổng
thời gian, chạy bất đồng bộ nên không chặn pipeline chính.

---

## 5. Kết luận rút ra từ số liệu này

1. **RAG (SHAP k-NN) hoạt động tốt trên hầu hết nhãn** — 80.7% tổng, 100% ở
   9/14 nhãn khi retrieval có đủ dữ liệu train.
2. **VLM không cộng thêm giá trị so với chỉ dùng RAG — nó làm giảm** (M6 =
   −15.7pp, McNemar p < 0.0001, không ca nào VLM sửa đúng lỗi RAG).
3. **Nơi RAG yếu sẵn (pool mỏng), VLM cũng không cứu được** — vấn đề nằm ở
   tầng retrieval, không phải tầng suy luận.
4. **Confidence nội tại (logprob) không dùng được để gating** — tương quan
   NGƯỢC với độ chính xác (AUROC 0.467). Confidence tự khai tương quan đúng
   chiều nhưng không phản ánh trạng thái thật của model.
5. **Định vị đóng góp phù hợp:** đo được error profile cụ thể của
   VLM-as-forensic-reporter trên NIDS, kèm bằng chứng định lượng (McNemar,
   per-label breakdown, phân tích định tính 2 ca lỗi) cho thấy attribution
   vẫn nên đến từ tầng truy hồi, không phải từ suy luận tự do của LLM.
