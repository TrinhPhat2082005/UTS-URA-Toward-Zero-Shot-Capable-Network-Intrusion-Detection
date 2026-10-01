# 03_splits_v2 — Chia lại tỷ lệ Benign:Attack theo thực tế CIC-IDS-2018

## Vì sao có bản v2 này

Bản gốc `03_splits/` (v1) dùng `BENIGN_CAP=100,000` — sau khi gộp 13 loại attack còn
lại (mỗi loại cap `ATTACK_CAP_PER_CLASS=20,000`) vào `train_benign_attack.csv`, tổng
Attack (132,613–152,560 dòng tùy fold) **vượt xa** Benign (chỉ 80,000 dòng sau 80%
split) — tỷ lệ Benign chỉ còn ~35-38%, **đảo ngược hoàn toàn** so với CIC-IDS-2018 gốc
(Benign chiếm 83.07% theo khảo sát nhiều paper dùng XGBoost/tree-based trên dataset
này — xem `zeroshot_benchmark/04_training/RESULTS.md`).

**Ngoài ra v1 còn có 1 data leakage đã phát hiện**: `train_benign_only.csv` (dùng
train AE-Only/SVDD-Only/CLAD) chứa **100%** Benign đã cap, nhưng `val_binary.csv` của
mỗi fold cũng lấy 20% Benign từ **cùng đúng pool đó** (do `train_test_split` xáo trộn
ngẫu nhiên toàn bộ Benign+Attack mỗi lần gọi `build_split_for_label()`). Kết quả: AE/
SVDD/CLAD encoder đã "thấy" trước chính những dòng Benign dùng để tune threshold và
đánh giá known-attack — threshold/eval trên Benign bị lạc quan giả tạo.

## v2 sửa 2 vấn đề này

1. **`BENIGN_CAP` 100,000 → 700,000** — đạt tỷ lệ Benign ~82-84% mỗi fold (sát mục
   tiêu 83.07% CIC-IDS-2018 gốc), thay vì ~35-38% như v1.
2. **Tách Benign thành 2 pool CỐ ĐỊNH, không chồng lấn, chỉ 1 lần duy nhất** cho toàn
   bộ pipeline:
   - `benign_train_pool` (80% của 700K = 560,000 dòng) → ghi ra `train_benign_only.csv`
     VÀ là toàn bộ phần Benign trong `train_benign_attack.csv` của MỌI fold.
   - `benign_val_pool` (20% còn lại = 140,000 dòng) → là toàn bộ phần Benign trong
     `val_binary.csv` của MỌI fold — **không bao giờ** được đưa vào bất kỳ bước train
     nào (AE/SVDD/CLAD/XGBoost).

## Cách build (không đọc lại raw parquet cho Attack)

Phần Attack (13 nhãn còn lại mỗi fold) **không đổi** so với v1 — v1 đã
`train_test_split(stratify=Label_Text)` đúng cách cho Attack rồi, không cần chia lại.
Script `build_v2_from_v1.py` tái sử dụng trực tiếp:

- `03_splits/by_label/<label>/test_zeroshot.csv` — copy nguyên vẹn sang v2 (100% attack
  thuần của nhãn bị giữ lại, không liên quan gì đến Benign, không cần đổi).
- `03_splits/by_label/<label>/train_benign_attack.csv` — **lọc bỏ** các dòng
  `Label_Text == "Benign"`, giữ nguyên phần Attack (13 nhãn còn lại), rồi **ghép** với
  `benign_train_pool` (Benign mới, 560,000 dòng, y hệt cho mọi fold).
- `03_splits/by_label/<label>/val_binary.csv` — tương tự, lọc bỏ Benign cũ, ghép với
  `benign_val_pool` (Benign mới, 140,000 dòng, y hệt cho mọi fold).

Nguồn Benign mới lấy từ `02_cleaning/clean_data/by_label/Benign.parquet` (raw đã làm
sạch, 10,564,771 dòng), sample lại 700,000 dòng bằng `random_state=42` (khác seed/lần
sample so với v1 — đây LÀ MỘT TẬP BENIGN MỚI, không phải tập con của 100K cũ).

## Output

```
03_splits_v2/
├── README.md                          # file này
├── build_v2_from_v1.py                # script build (đọc v1 + Benign.parquet, ghi v2)
├── split_manifest_v2.json             # audit: cap, tỷ lệ, checksum
├── train_benign_only.csv              # benign_train_pool, 560,000 dòng
└── by_label/<label>/
    ├── train_benign_attack.csv        # benign_train_pool (560K) + 13 attack cũ (không đổi)
    ├── val_binary.csv                 # benign_val_pool (140K) + 13 attack cũ (không đổi)
    └── test_zeroshot.csv              # copy nguyên vẹn từ v1, không đổi
```

## Không đụng vào

`03_splits/`, `04_training/`, và mọi artifacts/checkpoint bên trong giữ nguyên 100%,
không bị sửa/xóa. `03_splits_v2/` và `04_training_v2/` là 2 thư mục hoàn toàn độc lập.
