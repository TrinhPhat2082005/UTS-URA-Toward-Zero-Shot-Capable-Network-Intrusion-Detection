# 02_splits — split leave-one-attack-out (Protocol A)

Một script duy nhất, `build_splits.py`, thay cho hai bước cũ (`03_splits/build_leave_one_out_splits.py`
rồi `03_splits_v2/build_v2_from_v1.py`). Script cho ra **đúng từng byte** bộ split đang dùng để train
mọi checkpoint Protocol A (`zeroshot_benchmark/03_splits_v2/`).

## Chạy

```
python build_splits.py                            # đọc $UTS_BENCH/02_cleaning/clean_data/by_label
                                                  # ghi  $UTS_BENCH/03_splits_v2
python build_splits.py --clean-dir DIR --out-dir DIR
```

Cuối script so sha256 của 42 file theo fold với `split_manifest_reference.json` (manifest gốc, sinh
trên Kaggle ngày 2026-09-09) và in số file khớp.

## Output

```
03_splits_v2/
├── train_benign_only.csv              benign train pool, 560,000 dòng (train AE/SVDD/CLAD)
├── split_manifest.json                cap, số dòng, sha256 từng file
└── by_label/<held_out>/
    ├── train_benign_attack.csv        benign train pool + 80% của 13 attack còn lại
    ├── val_binary.csv                 benign val pool (140,000) + 20% của 13 attack còn lại
    └── test_zeroshot.csv              toàn bộ attack bị giữ lại (đã cap), không benign
```

## Các bước

1. Mỗi attack class lấy tối đa 20,000 dòng (seed 42); class ít hơn giữ nguyên.
2. Chia attack 80/20 cho từng fold bằng `train_test_split(stratify=Label_Text, seed=42)` trên
   [100K benign + 13 attack còn lại], đúng như bản v1. 100K benign này **chỉ** để tái tạo đúng
   flow attack nào rơi vào train/val (hàm xáo trộn cả tập gộp); nó bị bỏ ngay sau khi chia.
3. Benign: lấy 700,000 dòng mới (seed 42), chia **cố định** 560K / 140K, dùng chung cho mọi fold.
4. Mỗi fold ghép benign pool với phần attack. Phần attack đi qua một vòng ghi/đọc CSV trong bộ nhớ:
   bản v2 đã đọc lại file CSV của v1 bằng `pd.read_csv`, mà bộ parse float mặc định không giữ đúng
   chữ số cuối, nên split hiện tại mang dấu vết đó (giá trị lệch cỡ 1e-16).

## Vì sao benign có hai pool cố định

- Bản v1 dùng 100K benign: benign chỉ còn ~35–38% mỗi fold, ngược với CIC-IDS-2018 gốc (~83%).
- Bản v1 lấy benign của `val_binary.csv` từ cùng pool với `train_benign_only.csv`, nên AE/SVDD/CLAD đã
  thấy chính benign dùng để chọn ngưỡng và đánh giá (leak). 700K và hai pool không chồng lấn sửa cả hai.

## Kiểm chứng việc gộp

Pipeline cũ (v1 rồi v2, lấy từ git) và `build_splits.py` chạy trên cùng một bộ dữ liệu giả (14 nhãn,
kích thước 7–500 dòng, benign nhiều row group, cap thu nhỏ): 43/43 file trùng sha256.
Chưa chạy lại trên dữ liệu đầy đủ; dữ liệu đã làm sạch hiện có trong workspace là một lần chạy dở,
còn split gốc được tạo trên Kaggle.
