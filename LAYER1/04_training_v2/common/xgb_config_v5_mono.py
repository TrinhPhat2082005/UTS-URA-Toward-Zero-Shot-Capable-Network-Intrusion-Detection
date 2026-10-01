"""
Config XGBoost v5-mono — GIONG HET xgb_config_v4_new_data.py ve moi tham so cay,
CHI THEM 1 thu: bat rang buoc DON DIEU (monotone constraint) cho cac feature
augmented co ngu nghia "cang lon cang bat thuong".

=== VAN DE DANG GIAI QUYET ===
Tren data v2, 3 bien the (XGBoost-only / AE+XGBoost / SVDD+XGBoost) cho ket qua
zero-shot gan nhu y het nhau (ZS-F1 0.586 / 0.574 / 0.592) — augmentation KHONG
tao ra khac biet. Da dieu tra bang SHAP va per-label:

  1. SHAP BAC BO gia thuyet "XGBoost bo qua feature augmented":
     - AE+XGBoost (fold Bot): ae_recon_error XEP HANG 1 (SHAP 0.79 > Dst Port 0.78)
     - SVDD+XGBoost (fold Bot): 5/15 feature top la svdd_dist_*
     - SVDD+XGBoost (fold Infiltration): 6/15 feature top la svdd_dist_*
     => XGBoost DUNG RAT TICH CUC cac feature nay.

  2. Nhung per-label lai cho thay chung VO DUNG tren zero-shot:
     nhan              SVDD-Only   SVDD+XGB
     SSH_Bruteforce      1.0000      0.0001
     DDoS_LOIC_HTTP      0.9282      0.0016
     Infiltration        0.4692      0.0001
     DDOS_HOIC           0.6030      0.0040
     => CUNG mot svdd_dist lam dau vao. Dung truc tiep lam anomaly score thi bat
        duoc; dua qua XGBoost thi mat sach.

=== CO CHE THAT BAI (ly do can monotone constraint) ===
XGBoost la supervised learner, toi uu loss tren 13 lop attack DA BIET. No hoc ra
cac LUAT LIEN HOP kieu:

        svdd_dist_11 > 2.5 ?
         /            \
       Co            Khong -> benign
        |
   Dst Port in {80,443} ?
    /          \
  Co          Khong
   |            |
-> attack   -> BENIGN (0.08)   <-- LA GAY VAN DE

Mau SSH_Bruteforce zero-shot co d_min RAT CAO (SVDD biet la bat thuong) nhung
Dst Port=22, va trong 13 lop known khong co lop nao dung port 22 => roi vao la
"benign". Cay hoc dung theo tap train (mau port-22 + d_min cao trong train THAT SU
la benign/SSH hop le), nhung khai quat hoa sai sang lop thu 14.

Ban chat: quan he giua svdd_dist va P(attack) bi hoc thanh KHONG DON DIEU —
d_min tang co the lam proba GIAM, tuy nhanh.

=== MONOTONE CONSTRAINT LAM GI ===
monotone_constraints la rang buoc CUNG o muc thuat toan: voi feature danh dau +1,
XGBoost BI CAM tao bat ky split nao khien gia tri du doan giam khi feature do tang.
Khi xet split tren svdd_dist tai nguong theta, neu value(nhanh phai, d>theta) <
value(nhanh trai, d<=theta) thi split do bi LOAI khoi danh sach ung vien. Rang buoc
duoc truyen xuong toan bo cay con (qua lower_bound/upper_bound moi node), nen cay
o tren KHONG THE TON TAI.

=> Ep ranh gioi quyet dinh cua ban tang cuong KHAC HAN ban goc: khong con duoc phep
   hoc luat "d_min cao NHUNG cac feature khac trong quen => benign".

=== CHI RANG BUOC FEATURE CO NGU NGHIA DON DIEU ===
  - svdd sphere_dist / d_min : CO (cang xa vung benign => cang bat thuong)   -> +1
  - ae_recon_error           : CO (tai tao cang te => cang bat thuong)       -> +1
  - ae_latent_*              : KHONG (toa do trong khong gian nen, gia tri
                               cao/thap deu co the la benign)                -> 0
  - 78 feature CICFlowMeter goc : KHONG rang buoc (giu tu do nhu cu)         -> 0

=== TUPLE monotone_constraints DUNG O DAU ===
KHONG dat gia tri co dinh trong file config nay, vi do dai tuple phai bang DUNG
so cot dau vao, ma 3 script co so cot KHAC NHAU:
    train_xgboost_only.py : 78 cot (khong co augmented -> khong rang buoc gi)
    train_ae_xgboost.py   : 95 cot (78 goc + 16 latent + 1 recon_error)
    train_svdd_xgboost.py : 91 cot (78 goc + 12 sphere_dist + 1 d_min, k_in=12)
Ngoai ra k_in cua SVDD chi biet duoc luc RUNTIME (dem so file encoder_*.pt trong
checkpoint), nen khong the hardcode. => Moi script tu dung tuple qua ham
build_monotone_constraints() ben duoi.

=== DANH DOI CAN LUONG TRUOC ===
  - KA-F1 co the giam nhe: rang buoc thu hep khong gian ham, model mat kha nang
    bieu dien mot so quan he CO THAT trong train. Neu KA-F1 tut duoi ~0.92 thi
    can can nhac lai.
  - KA-FPR co the tang: mau benign co d_min cao (benign bat thuong nhung hop le)
    gio khong the duoc "cuu" bang feature goc nua => de bi bao nham. Day la mat
    trai cua viec buoc model tin vao tin hieu one-class.
"""
from __future__ import annotations

XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="aucpr",
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    # --- Chong Overfitting ---
    gamma=0.2,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    reg_alpha=0.1,
    # --- System ---
    tree_method="hist",   # monotone_constraints yeu cau hist/approx/exact (hist OK)
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)

# Giu nguyen divisor=2.0 nhu xgb_config_v4_new_data.py de so sanh SACH: chi co
# monotone constraint la bien so thay doi duy nhat so voi ban v4-new-data.
XGB_SCALE_POS_DIVISOR = 2.0

XGB_PRED_THRESHOLD = 0.5  # du phong; cac script tune threshold rieng tren train

# Bat/tat rang buoc don dieu. Dat False de tai lap dung ket qua v4-new-data.
XGB_MONOTONE_AUGMENTED = True


def build_monotone_constraints(n_base: int, n_augmented_monotone: int,
                               n_augmented_free: int = 0) -> tuple[int, ...] | None:
    """Dung tuple monotone_constraints theo dung thu tu cot cua tung script.

    Thu tu cot phai khop CHINH XAC voi np.concatenate() trong script goi:
        [n_base feature goc | n_augmented_free feature tu do | n_augmented_monotone]

    Args:
        n_base: so feature CICFlowMeter goc (78) — khong rang buoc.
        n_augmented_monotone: so feature augmented CO ngu nghia don dieu tang
            (svdd: k_in sphere_dist + 1 d_min; ae: 1 recon_error).
        n_augmented_free: so feature augmented KHONG co ngu nghia don dieu
            (ae: 16 latent). Mac dinh 0.

    Returns:
        Tuple do dai (n_base + n_augmented_free + n_augmented_monotone),
        hoac None neu XGB_MONOTONE_AUGMENTED=False (XGBoost bo qua tham so).
    """
    if not XGB_MONOTONE_AUGMENTED:
        return None
    return tuple([0] * n_base + [0] * n_augmented_free + [1] * n_augmented_monotone)
