"""
Config XGBoost DUNG CHUNG cho ca 3 bien the tren DATA MOI (03_splits_v2, ty le
Benign:Attack thuc te ~83% benign, khong con leak train/val benign) — xem
common/xgb_config_v4.py (goc, dung cho 04_training/ v1) de doi chieu.

Giong het xgb_config_v4.py ve kien truc cay (n_estimators=500, max_depth=6,
learning_rate=0.03, regularization...), CHI KHAC 1 diem: XGB_SCALE_POS_DIVISOR = 2.0
(khoi phuc lai viec chia doi scale_pos_weight, thay vi = 1.0 nhu xgb_config_v4.py).

Ly do khac biet: xgb_config_v4.py bo chia doi (dat = 1.0) vi luc do ty le benign:attack
trong 03_splits (v1) da GAN 1:1 (scale_pos ~0.66-0.75 — do BENIGN_CAP=100K qua nho so
voi ATTACK_CAP_PER_CLASS=20K x 13 nhan con lai), nen chia doi se day model LECH XA khoi
vung ty le ma cac paper tham khao nham toi. Nhung voi 03_splits_v2 (BENIGN_CAP=700K,
ty le benign that ~82-84%), scale_pos_weight goc (tinh dong = so benign / so attack cua
tung fold) gio LON (~4-5, vi benign chiem da so that su) — dung nguyen scale_pos_weight
nay se ep XGBoost uu tien lop attack (positive class) manh hon nhieu so voi ty le 1:1 ma
cac paper tham khao (Songma 2023, v.v.) huong toi. Chia doi (XGB_SCALE_POS_DIVISOR=2.0)
dua scale_pos_weight ve muc trung gian (~2-2.5) — giam bot muc do uu tien attack so voi
ty le that 4-5:1, gan hon voi vung 1:1-2:1 ma cac paper dung XGBoost/tree-based tren
CIC-IDS-2018/2017 thuong nham toi, theo yeu cau nguoi dung 2026-09-09 khi chuyen sang
train tren data v2.
"""
from __future__ import annotations

XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="aucpr",           # Toi uu Precision-Recall thay vi logloss
    n_estimators=500,              # Tang len 500 vi da co early stopping bao ve
    learning_rate=0.03,
    max_depth=6,
    # --- Chong Overfitting ---
    gamma=0.2,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=2.0,                # L2 reg
    reg_alpha=0.1,                 # L1 reg
    # --- System ---
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,      # Dung som neu sau 50 cay khong tien bo
)

# scale_pos_weight thuc te = scale_pos / XGB_SCALE_POS_DIVISOR (scale_pos tinh dong
# theo ti le benign/attack cua tung fold trong moi script). Dat = 2.0 (khoi phuc chia
# doi) — xem giai thich o docstring dau file: data v2 co ty le benign that ~83%, nen
# scale_pos_weight goc lon hon nhieu so voi v1, can chia bot de khong lech qua xa
# vung ty le 1:1-2:1 ma cac paper tham khao nham toi.
XGB_SCALE_POS_DIVISOR = 2.0

XGB_PRED_THRESHOLD = 0.5  # gia tri du phong; cac script hien tune threshold rieng
                          # theo best_f1_threshold() tren train_benign_attack, khong
                          # dung co dinh nay.
