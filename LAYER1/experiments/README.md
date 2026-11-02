# Thí nghiệm Layer 1 (RQ1, RQ2)

Đánh giá Layer 1 trên Protocol A (14 fold leave-one-attack-out). Code ở đây **không train lại**:
nó nạp checkpoint đã có, chấm điểm, rồi áp ngưỡng theo ngân sách FPR. Ngưỡng in trong các script
`04_training_v2/train_*.py` (F1-max trên val) chỉ để tham khảo, không phải ngưỡng của paper.

## Chạy

| Notebook | Kết quả |
|---|---|
| `layer1_results.ipynb` | Mỗi cell một model, in một dòng của bảng RQ1/RQ2 trên slide và bảng từng fold của model đó. Cell cuối ghi `output/rq1_rq2_baselines/final/`: `table_main.csv`, `curve.csv`, `thresholds_main.csv`, `final_results.md`, `manifest.json` |
| `layer1_extras.ipynb` | `verify.md` (tái hiện đánh giá của từng người train), so sánh hai run train, `rq1_rq2_layer1/layer1_tables.md` (6 model của paper, ngưỡng riêng từng fold), hình `final/zs_vs_alpha.pdf`, latency `baseline_latency.json` |

Chạy `layer1_results.ipynb` trước, vì hình trong `layer1_extras.ipynb` đọc `final/curve.csv`.
Mặc định điểm được đọc từ cache trong `output/`. Đặt `RECOMPUTE = True` để chấm lại từ checkpoint,
mất từ vài phút đến vài chục phút mỗi model; OptiForest mất khoảng 25 phút. Chạy bằng conda env `ai_env`.

## Code (`layer1_eval/`)

| File | Nội dung |
|---|---|
| `paths.py` | Đường dẫn dữ liệu, checkpoint, output (biến môi trường bên dưới) |
| `data.py` | Đọc split: 560K benign train (chung 14 fold), `val_binary` và `test_zeroshot` của từng fold |
| `scorers.py` | Checkpoint → `score(X)` cho 16 model (điểm cao = attack), tiền xử lý đúng như lúc train |
| `scores.py` | 4 phần điểm mỗi fold (`tb` benign train, `vb`/`va` benign/attack của val, `zs` test zero-shot), có cache |
| `metrics.py` | Ngưỡng theo FPR, Acc / Precision / Recall / F1 / FPR / ZS-Acc, pAUC[0, 3%] |
| `tables.py` | Cấu hình bảng (`ROWS`, `RULE`, `ALPHA_TABLE`, `ALPHA_GRID`), `run(role)`, `write_final(results)` |
| `extras.py` | `latency`, `plot_zs_vs_alpha`, `perfold_tables`, `verify_reproduction`, `checkpoint_runs` |

## Ngưỡng và chỉ số

- Ngưỡng của fold f: τ_f = Q₁₋α(điểm trên 560K benign train), `np.quantile(..., method="higher")`.
  Một flow là alert nếu điểm ≥ ngưỡng. Val và test không tham gia đặt ngưỡng.
- `meanfold` (bảng trên slide, α = 0.5%): mọi fold dùng chung τ = trung bình 14 giá trị τ_f.
  Lấy trung bình làm FPR thực tế hơi cao hơn α với đa số model, và cao hơn nhiều với SVM và Random Forest,
  vì thang điểm của hai model này khác nhau giữa các fold.
- `perfold` (`layer1_tables.md`, α = 0.4% và 1.5%): fold f dùng τ_f của chính nó.
- Known-attack (Acc, Precision, Recall, F1, FPR) tính trên `val_binary` (benign + 13 loại đã biết);
  ZS-Acc tính trên `test_zeroshot` (loại bị giữ lại). Mọi chỉ số là trung bình 14 fold.
- pAUC[0, 3%] là diện tích dưới đường ZS-Acc theo FPR thực tế trên benign val, với FPR ∈ [0, 3%],
  chia cho 3%. Chỉ số này không phụ thuộc ngưỡng; mức ngẫu nhiên là 0.015.

## Checkpoint của từng dòng

`tables.ROWS` chọn checkpoint cho từng dòng (quyết định ngày 2026-10-05):

- XGBoost-only, AE-only và AE-XGBoost (ours) lấy từ run của paper (`UTS_LAYER1_ART`).
- SVDD-XGBoost (ours) và mọi baseline khác lấy từ run của baseline study (`UTS_BASELINE_ART`).
- XGBoost-only là cùng một file ở cả hai run.

`extras.checkpoint_runs()` so sánh hai run của sáu model có ở cả hai. Train lại encoder AE/SVDD trên máy
khác cho trọng số khác (GPU không tất định; K-Means cũng khác giữa scikit-learn 1.8 và 1.9), và ZS-Acc
của các hybrid khi đó dịch tới khoảng 0.05, chủ yếu qua vài fold như DDOS_attack_HOIC.
Code train của baseline study nằm ở `LAYER1/baseline_study/`.

## Các model được dựng lại

- **KNN** (`PyTorchKNNClassifier`, pickle dưới `__main__`): 50K flow train đã scale, k = 5, vote theo 1/khoảng cách.
- **VAE**: kiến trúc suy từ tên và shape tham số trong `vae_best.pt`. Điểm = MSE tái tạo + (0.1/78) · KL.
- **Deep SVDD** (Ruff et al., ICML 2018), từ `deep_svdd_paper/oneclass_deep_svdd.pt`: encoder không bias
  78-256-128-64, LeakyReLU 0.1, tâm cố định. Đây là model khác với **Mixture Deep SVDD** (`svdd_only`) của paper:
  bản Mixture chia benign bằng K-means và fit một sphere cho mỗi cụm, để khoảng cách tới các sphere làm feature
  cho XGBoost.
- **OptiForest** (Xiang et al., IJCAI 2023) được chấm bằng code của tác giả. Repo của tác giả không có license,
  nên code đó không được chép vào đây; trỏ `UTS_OPTIFOREST` tới thư mục chứa `detectors/`.
  Điểm = −`decision_function`. Điểm benign train và benign val lấy từ file người train đã lưu (chấm lại
  8.000 flow khớp tới 1e-16); chỉ chấm lại các flow attack, mỗi flow khác nhau chấm một lần.

`extras.verify_reproduction()` áp ngưỡng riêng của từng người train lên điểm đã chấm lại.
Mọi dòng phải ra `yes` thì bảng mới dùng được. Riêng Deep SVDD không có `metrics.json` cạnh checkpoint,
nên không kiểm theo cách này.

## Cache và tái lập

- Điểm của các model của paper nằm trong `output/rq1_rq2_layer1/sweep_all_scores.npz` (một file chung;
  RQ3b của LAYER2 cũng đọc file này) và giữ kiểu dữ liệu gốc. Điểm của baseline study nằm trong
  `output/rq1_rq2_baselines/scores_<model>.npz`, ép float32. Đổi kiểu dữ liệu có thể làm lệch các flow nằm
  đúng trên ngưỡng.
- `thresholds_main.csv` lưu ngưỡng chính xác (`repr`) của mọi model ở mọi fold, kèm FPR, recall và ZS-Acc của fold.
  `manifest.json` lưu phiên bản code, phiên bản thư viện, MD5 của mọi checkpoint và mọi file điểm.
  Checkpoint không nằm trong git; hãy so MD5 trước khi chạy lại.
- Chấm lại 8.192 flow đầu (đúng một batch như lúc tạo cache) cho điểm trùng khớp cache. Chấm một tập con có kích
  thước khác thì encoder cho sai số float rất nhỏ. Sai số này làm vài phần trăm flow benign của các hybrid SVDD
  dịch qua điểm cắt của XGBoost (hist cắt đúng tại giá trị của flow train), làm p(x) đổi tới khoảng 0.04.
- Môi trường: python 3.11, numpy 2.4.6, scikit-learn 1.8.0, xgboost 3.2.0, torch 2.5.1+cu121, lightgbm 4.7.0.
  Trên máy 16 GB, đặt `OPENBLAS_NUM_THREADS=2` và `CUDA_VISIBLE_DEVICES=`; cell đầu của notebook đã đặt sẵn.

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `UTS_WS` | cha của repo (`workspace/`) | Thư mục gốc workspace |
| `UTS_BENCH` | `$UTS_WS/zeroshot_benchmark` | Split `03_splits_v2` và checkpoint Protocol A |
| `UTS_LAYER1_CODE` | `LAYER1/04_training_v2` | `models/`, `common/` của Layer 1 |
| `UTS_LAYER1_ART` | `$UTS_BENCH/04_training_v2/artifacts` | Checkpoint 14 fold của paper |
| `UTS_BASELINE_ART` | `$UTS_WS/baselines_friends/artifacts` | Checkpoint của baseline study |
| `UTS_OPTIFOREST` | `<cha của workspace>/isolation_forest` | Thư mục chứa package `detectors/` của OptIForest |
| `UTS_L1_RESULTS` | `LAYER1/output` | Nơi ghi kết quả |
