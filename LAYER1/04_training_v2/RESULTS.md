# Zero-Shot Benchmark v2 — Kết quả tổng hợp (data thực tế 83% Benign)

Nguồn: `zeroshot_benchmark/04_training_v2/artifacts/<variant>/metrics.json`, đo trên bộ
split leave-one-attack-out (14 fold, mỗi fold giữ lại 1 trong 14 loại tấn công
CIC-IDS-2018), dùng data `03_splits_v2/` (không phải `03_splits/` gốc — xem
`03_splits_v2/README.md` để biết đầy đủ lý do redesign).

- **Zero-shot (ZS)**: đo trên `test_zeroshot.csv` — 100% mẫu của nhãn bị giữ lại, đo
  recall thuần trên tấn công chưa từng thấy trong huấn luyện.
- **Known-attack (KA)**: đo trên `val_binary.csv` — Benign + 13 loại tấn công còn lại
  (đã biết), đo được cả FPR/Precision thật vì có Benign.
- Tất cả số liệu là **macro-average qua 14 fold**.

## 0. Vì sao có v2 — khác gì so với v1 (`04_training/`)

| | v1 (`03_splits/`) | v2 (`03_splits_v2/`) |
|---|---|---|
| BENIGN_CAP | 100,000 | 700,000 |
| Tỷ lệ Benign:Attack trong train | ~35-38% Benign (**lệch ngược** so với thực tế) | **82-84% Benign** (khớp CIC-IDS-2018 gốc ~83%) |
| AE/SVDD encoder train/val benign | **CÙNG 1 pool** (leak — encoder "thấy" benign val) | 2 pool **cố định, không chồng lấp** (fix leak) |
| Attack data | Stratified split gốc | **Tái sử dụng nguyên vẹn** từ v1 (chỉ lọc lại Benign) |

v1 vô tình đảo ngược tỷ lệ thật của CIC-IDS-2018 (dataset gốc ~83% Benign) do
`BENIGN_CAP=100K` quá nhỏ so với `ATTACK_CAP_PER_CLASS=20K × 13 nhãn còn lại`. v2 sửa
cả hai vấn đề: tỷ lệ lớp và data leak khi train AE/SVDD trên benign.

## 1. Bảng tổng hợp chính (macro-avg qua 14 fold)

| Model | ZS-F1 | ZS-Recall | ZS-Precision | KA-F1 | KA-Recall | KA-Precision | KA-FPR |
|---|---|---|---|---|---|---|---|
| XGBoost-only | 0.5860 | 0.5520 | 0.9286 | 0.9347 | 0.8960 | 0.9773 | 0.0042 |
| AE-Only | 0.6253 | 0.5504 | 1.0000 | 0.4967 | 0.4134 | 0.6240 | 0.0500 |
| AE+XGBoost | 0.5742 | 0.5358 | 0.9286 | 0.9346 | 0.8933 | 0.9802 | 0.0036 |
| SVDD-Only (K=10) | **0.8875** | **0.8535** | 1.0000 | 0.3148 | 0.7442 | 0.1998 | **0.6027** |
| SVDD+XGBoost | 0.5918 | 0.5596 | 0.9286 | 0.9348 | 0.8943 | 0.9794 | 0.0038 |
| CLAD | 0.5746 | 0.5278 | 0.9286 | 0.9332 | 0.8903 | 0.9807 | 0.0035 |

**Ghi chú:**
- **Ba biến thể XGBoost gần như y hệt nhau** (ZS-F1 0.574–0.592) — feature augmentation
  (AE latent/recon_error, SVDD sphere-dist/d_min) **không tạo ra khác biệt đáng kể** trên
  data v2, trái với kỳ vọng ban đầu. Xem mục 4 để biết điều tra chi tiết nguyên nhân.
- **AE+XGBoost thậm chí thấp hơn XGBoost-only** (0.5742 < 0.5860) — augmentation ở đây
  phản tác dụng nhẹ, không chỉ "không giúp ích".
- **SVDD-Only đạt ZS-recall cao nhất (0.8535)** trong toàn bộ 5 model nhưng **KA-FPR =
  0.6027** — báo động nhầm ~60% lưu lượng benign thật, không khả dụng vận hành thực tế.
- **AE-Only có FPR thấp hơn nhiều (0.05)** nhưng đây là do thiết kế (ngưỡng chọn theo
  FPR budget cố định 5%, xem mục 5), không so sánh trực tiếp được với SVDD-Only (ngưỡng
  chọn theo F1-max, không giới hạn FPR).
- **CLAD xếp thấp nhất trong 4 model dùng chung tiêu chí ngưỡng F1-max** (ZS-Recall
  0.5278, so với XGBoost-only 0.5520, AE+XGBoost 0.5358, SVDD+XGBoost 0.5596) — nhưng
  khoảng cách rất nhỏ (~0.01–0.03), không phải chênh lệch lớn. Điểm bất ngờ nhất:
  `SSH_Bruteforce` — SVDD-Only (one-class thuần) đạt 1.0000 nhưng CLAD chỉ 0.0003, gần
  như thất bại giống hệt 3 biến thể XGBoost (0.0001), dù CLAD cũng dùng cơ chế khoảng
  cách-tới-centroid tương tự SVDD. Khả năng do CLAD được train trên benign+attack
  (không phải benign-only như SVDD), khiến ranh giới học được gần với supervised hơn là
  thuần one-class. Ngược lại, CLAD vượt trội rõ trên `SQL_Injection` (0.7857 — cao nhất
  trong 4 model) và `Infiltration` (0.0022 — thấp nhưng vẫn cao hơn cả 3 biến thể
  XGBoost).

## 2. So sánh với v1 (cùng kiến trúc, data khác)

| Model | ZS-F1 v1 | ZS-F1 v2 | Δ | KA-FPR v1 | KA-FPR v2 | Δ |
|---|---|---|---|---|---|---|
| XGBoost-only | 0.7493 | 0.5860 | −0.163 | 0.0309 | 0.0042 | −0.027 |
| AE+XGBoost | 0.7824 | 0.5742 | −0.208 | 0.0329 | 0.0036 | −0.029 |
| SVDD+XGBoost | 0.8080 | 0.5918 | −0.216 | 0.0295 | 0.0038 | −0.026 |
| SVDD-Only | 0.9614 | 0.8535 | −0.108 | 0.6100 | 0.6027 | −0.007 |
| CLAD | 0.7105 | 0.5746 | −0.136 | 0.0152 | 0.0035 | −0.012 |

Cả 3 biến thể XGBoost **giảm mạnh zero-shot** khi chuyển sang data v2, nhưng đồng thời
**KA-FPR giảm rất nhiều** (~7-8×, từ ~0.03 xuống ~0.004) — đây là đánh đổi hợp lý: khi
benign chiếm đa số thực tế (83%), model buộc phải "cẩn trọng" hơn để không báo động
tràn lan, và ranh giới quyết định học được ít thiên vị attack hơn. Zero-shot recall cao
ở v1 một phần là ảo do tỷ lệ lớp bị đảo ngược khiến model "quen" phân loại attack.

## 3. Per-label Zero-Shot Recall (6 model)

| Nhãn | XGBoost-only | AE+XGBoost | SVDD+XGBoost | AE-Only | SVDD-Only | CLAD |
|---|---|---|---|---|---|---|
| Bot | 0.0000 | 0.0003 | 0.0000 | 0.0063 | 0.0726 | 0.0000 |
| Brute_Force_Web | 0.4739 | 0.4559 | 0.4685 | 0.4775 | **0.9586** | 0.4793 |
| Brute_Force_XSS | 0.9649 | 0.7982 | 0.9518 | 0.5000 | 0.9956 | 0.6623 |
| DDOS_attack_HOIC | 0.0168 | 0.0000 | 0.0040 | 0.1484 | **0.6030** | 0.0123 |
| DDOS_attack_LOIC_UDP | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| DDoS_attacks_LOIC_HTTP | 0.0027 | 0.0019 | 0.0016 | 0.4335 | **0.9282** | 0.0016 |
| DoS_attacks_GoldenEye | 0.9963 | 0.9994 | 0.9919 | 0.7393 | 0.9962 | 0.9026 |
| DoS_attacks_Hulk | 0.9919 | 0.9877 | 0.9569 | 0.1477 | 0.9993 | 0.8477 |
| DoS_attacks_SlowHTTPTest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| DoS_attacks_Slowloris | 0.7690 | 0.7696 | 0.7696 | 0.9864 | 0.9855 | 0.6953 |
| FTP_BruteForce | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Infiltration | 0.0001 | 0.0003 | 0.0001 | 0.0876 | 0.4692 | **0.0022** |
| SQL_Injection | 0.5119 | 0.4881 | 0.6905 | 0.1786 | 0.9405 | **0.7857** |
| SSH_Bruteforce | 0.0001 | 0.0001 | 0.0001 | 1.0000 | **1.0000** | 0.0003 |
| **Macro-avg** | **0.5520** | **0.5358** | **0.5596** | **0.5504** | **0.8535** | **0.5278** |

**3 nhãn "tàng hình"** — cả 6 model đều gần như bỏ lỡ hoàn toàn: `Bot` (traffic âm thầm),
`DDoS_attack_HOIC`, `DDoS_attacks_LOIC_HTTP`. Đây là các attack có đặc trưng lưu lượng
gần giống benign, khó phân biệt cho cả supervised (XGBoost, CLAD) lẫn one-class (AE/SVDD).
`Infiltration` cũng khó nhưng CLAD (0.0022) và SVDD-Only (0.4692) vẫn nhỉnh hơn hẳn nhóm
XGBoost-augmented (~0.0001–0.0003).

**Nghịch lý nổi bật nhất**: `SSH_Bruteforce` — SVDD-Only đạt **1.0000** nhưng
SVDD+XGBoost chỉ **0.0001** (chênh lệch ×10,000), và ngay cả **CLAD (0.0003)** — vốn
cũng dùng cơ chế khoảng cách-tới-centroid tương tự SVDD — cũng thất bại gần như hoàn
toàn trên nhãn này. Cùng một loại tín hiệu one-class, chỉ SVDD-Only (huấn luyện thuần
benign-only, không tiếp xúc attack) giữ được năng lực phát hiện; mọi kiến trúc có tiếp
xúc với attack trong quá trình huấn luyện (XGBoost-augmented, CLAD) đều mất tín hiệu
này. Xem mục 4.

## 4. Điều tra: vì sao Feature-Augmentation không tạo khác biệt trên v2

### 4.1. Giả thuyết "XGBoost bỏ qua feature augmented" — BỊ BÁC BỎ bởi SHAP

- `AE+XGBoost` (fold Bot): `ae_recon_error` **xếp hạng 1** SHAP (0.79, vượt cả `Dst Port`
  0.78).
- `SVDD+XGBoost` (fold Bot): 5/15 feature top là `svdd_dist_*`.
- `SVDD+XGBoost` (fold Infiltration): 6/15 feature top là `svdd_dist_*`.

XGBoost **dùng rất tích cực** các feature augmented — vấn đề không phải "bỏ qua".

### 4.2. Cơ chế thất bại: luật liên hợp nghịch biến

XGBoost là supervised learner, tối ưu trên 13 lớp attack **đã biết**. Nó học ra các luật
liên hợp dạng: *"svdd_dist cao NHƯNG Dst Port quen thuộc → benign"*. Với attack thứ 14
(chưa từng thấy), mẫu có `d_min` rất cao (SVDD biết là bất thường) nhưng rơi vào nhánh
cây sai vì các feature gốc đi kèm không khớp pattern đã học → bị phân loại benign.

### 4.3. Thử sửa bằng Monotonic Constraint — cải thiện rất nhỏ, không đủ

Ép ràng buộc đơn điệu (`monotone_constraints`, XGBoost API): feature augmented tăng →
P(attack) không được giảm. Xem `common/xgb_config_v5_mono.py`,
`train_ae_xgboost_mono.py`, `train_svdd_xgboost_mono.py`.

| Model | ZS-F1 base | ZS-F1 mono | Δ | KA-F1 | KA-FPR |
|---|---|---|---|---|---|
| AE+XGBoost | 0.5742 | **0.6060** | +0.0318 | 0.9348 (≈base) | 0.0037 (≈base) |
| SVDD+XGBoost (ràng buộc 13 cột) | 0.5918 | 0.5945 | +0.0027 | 0.9347 | 0.0038 |

KA-F1/KA-FPR **không hề suy giảm** — ràng buộc không gây hại. Nhưng **4 nhãn khó nhất
vẫn hoàn toàn bất động**: `SSH_Bruteforce` (0.0001→0.0001), `DDoS_LOIC_HTTP`
(0.0016→0.0019), `Infiltration` (0.0001→0.0003), `DDOS_HOIC` (0.0040→0.0073).

### 4.4. Sửa "lỗi ngữ nghĩa" của monotone constraint — kết quả tệ hơn

Phát hiện: ràng buộc `+1` trên **toàn bộ 12 `sphere_dist`** (khoảng cách tới từng tâm
cụm riêng lẻ) là sai ngữ nghĩa — một benign hợp lệ thuộc cụm *j* sẽ có `sphere_dist_k`
**cao tự nhiên** với mọi *k ≠ j* (chỉ vì nó gần cụm *j*, xa các cụm khác), không phải vì
bất thường. Sửa: chỉ ràng buộc `d_min` (= min của 12 sphere_dist — khoảng cách tới cụm
**gần nhất**, mới có ngữ nghĩa đơn điệu đúng), để 12 `sphere_dist` tự do.

Kết quả (`train_svdd_xgboost_mono2.py`, `artifacts/svdd_xgboost_mono2/`):

| Cấu hình | ZS-F1 | ZS-Recall | threshold_mean |
|---|---|---|---|
| base (không ràng buộc) | 0.5918 | 0.5596 | 0.5636 |
| mono (ràng buộc 13 cột — "sai" ngữ nghĩa) | 0.5945 | 0.5650 | 0.5617 |
| mono2 (chỉ ràng buộc d_min — "đúng" ngữ nghĩa) | **0.5863** | **0.5552** | 0.5797 |

**Kết quả ngược kỳ vọng**: cấu hình "đúng ngữ nghĩa hơn" (mono2) cho ZS-F1 **thấp hơn**
cả cấu hình "sai" (mono) lẫn base. 4 nhãn khó nhất vẫn bất động ở **cả 3 cấu hình**.

### 4.5. Kết luận

Chẩn đoán "ràng buộc trên `sphere_dist` riêng lẻ sai ngữ nghĩa" **đúng về mặt logic**
nhưng **không phải nguyên nhân chính** khiến augmentation thất bại — bằng chứng là sửa
đúng ngữ nghĩa không cải thiện, thậm chí giảm nhẹ. Nguyên nhân sâu hơn: monotonic
constraint chỉ ràng buộc **hướng** của quan hệ, không ràng buộc được **biên độ** ảnh
hưởng lên quyết định cuối. Trên tập train, 13 lớp attack đã biết được nhận diện chủ yếu
qua 78 feature CICFlowMeter gốc; feature augmented chỉ đóng vai trò phụ trợ — cây học
đúng điều đó và gán trọng số nhỏ, nên khi gặp lớp thứ 14 (chỉ có augmented feature làm
tín hiệu), biên độ proba không đủ vượt ngưỡng trung bình (~0.56–0.58), bất kể ràng buộc
đơn điệu đặt đúng hay sai chỗ nào.

**Feature-level augmentation về bản chất không bảo toàn được năng lực one-class khi đưa
qua supervised loss** — đây là giới hạn cấu trúc, không phải lỗi cài đặt hay thiếu tinh
chỉnh. Hai hướng khả thi còn lại (chưa triển khai, xem
`paper_outline_latex/augmentation_limitation_evidence.note`):

1. **Score-level OR-fusion**: `alert = (XGB_proba ≥ t_xgb) OR (svdd_score ≥ t_svdd)` —
   giữ `d_min` làm kênh quyết định độc lập, không để supervised loss chạm vào, bảo toàn
   biên độ tín hiệu one-class. Không cần train lại model, SHAP vẫn chính xác.
2. **Ép biên độ**: biến đổi phi tuyến feature augmented (`exp(d_min/τ)` hoặc
   percentile-rank so với phân bố benign) để vùng bất thường có giá trị số vượt trội,
   buộc cây phải tách nhánh ở đó.

## 5. Ghi chú phương pháp

- **XGBoost config**: `common/xgb_config_v4_new_data.py` — kiến trúc giống v1
  (`n_estimators=500, max_depth=6, learning_rate=0.03`, regularization đầy đủ), CHỈ khác
  `XGB_SCALE_POS_DIVISOR=2.0` (khôi phục chia đôi `scale_pos_weight` — v1 dùng 1.0 vì tỷ
  lệ lớp gần 1:1, nhưng v2 có tỷ lệ benign thật ~5:1 nên cần chia bớt để không lệch quá
  xa vùng 1:1–2:1 mà các paper tham chiếu nhắm tới). Đã thử nghiệm cả divisor=1.0 và
  divisor=2.0 — divisor=2.0 thắng nhẹ ở mọi chỉ số trên cả 3 biến thể XGBoost (chênh lệch
  ~0.005-0.015 F1), không đáng kể nhưng nhất quán.
- **Threshold-averaging**: cả 3 biến thể XGBoost tune threshold theo F1-max trên
  `train_benign_attack.csv` (không phải `val_binary.csv`, để tránh leak — val là tập
  dùng để BÁO CÁO known-attack metrics), sau đó lấy **trung bình cộng** của 14 threshold
  riêng lẻ thành 1 threshold chung, dùng lại để đánh giá toàn bộ 14 fold (mô phỏng deploy
  thực tế với 1 ngưỡng cố định).
- **AE-Only threshold**: percentile 95 của recon-error trên benign tại `val_binary.csv`
  của 1 fold đại diện (`Bot`) — tương đương ép **FPR budget cố định 5%**.
- **SVDD-Only threshold**: F1-max trên PR-curve của `val_binary.csv` cùng fold đại diện —
  KHÔNG có ràng buộc FPR, nên chọn ngưỡng lỏng hơn nhiều (KA-FPR=0.60).
  ⚠️ **AE-Only và SVDD-Only dùng 2 tiêu chí ngưỡng khác nhau — ZS-recall của chúng KHÔNG
  so sánh trực tiếp công bằng được.** Nếu AE-Only được phép cùng FPR budget với
  SVDD-Only, ZS-recall của nó nhiều khả năng sẽ cao hơn đáng kể.
- **SVDD K-Means**: `K_CHOSEN=10` (ép `MIN_K_ALLOWED=6` — argmax(silhouette) không giới
  hạn ban đầu chọn nhầm K=2 do nhiễu ngẫu nhiên trên phân bố benign 560K dòng, không phản
  ánh cấu trúc cluster thật; xem lịch sử điều tra trong code `train_svdd_only.py`).
  `k_in=12` sau khi tách cluster lớn nhất thành 3 sub-cluster và loại các cụm/sub-cụm nhỏ.

## 6. File liên quan

```
04_training_v2/
├── common/
│   ├── xgb_config_v4_new_data.py    # config chính (divisor=2.0)
│   └── xgb_config_v5_mono.py        # + monotone constraint
├── train_xgboost_only.py            # baseline, không augment
├── train_ae_only.py / train_svdd_only.py
├── train_ae_xgboost.py / train_svdd_xgboost.py
├── train_ae_xgboost_mono.py / train_svdd_xgboost_mono.py   # + monotone (13 cột)
├── train_svdd_xgboost_mono2.py                              # + monotone (chỉ d_min)
├── train_clad.py                    # DA train xong 14/14 fold tren v2 (early-stop
│                                       + mid-fold resume, tung fold checkpoint duoc
│                                       dong bo ve local ngay khi hoan tat)
└── artifacts/
    ├── {ae_only,svdd_only,xgboost_only,ae_xgboost,svdd_xgboost,clad}/
    ├── {ae_xgboost,svdd_xgboost}_mono/, svdd_xgboost_mono2/
    ├── {xgboost_only,ae_xgboost,svdd_xgboost}_divisor1_backup/   # divisor=1.0, tham khảo
    └── MONO_VERSION.note            # ghi chú chi tiết các bản _mono
```

Phân tích sâu hơn và hướng đi tiếp theo:
`paper_outline_latex/augmentation_limitation_evidence.note`
