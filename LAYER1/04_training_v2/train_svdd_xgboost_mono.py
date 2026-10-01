"""
[BAN _mono] Train SVDD+XGBoost VOI RANG BUOC DON DIEU tren feature SVDD.

Khac ban goc train_svdd_xgboost.py DUNG 3 diem (moi thu khac giu nguyen 100%
de so sanh sach — cung data v2, cung encoder SVDD, cung threshold-averaging):
  1. Import config tu common/xgb_config_v5_mono.py (thay vi xgb_config_v4_new_data.py)
  2. Them monotone_constraints: k_in sphere_dist + d_min bi ep DON DIEU TANG
     (svdd_dist tang => P(attack) khong duoc giam). 78 feature goc giu tu do.
  3. Ghi ra artifacts/svdd_xgboost_mono/ (khong de len ket qua cu)

LY DO: tren data v2, SVDD+XGBoost (ZS-F1 0.592) gan nhu y het XGBoost-only
(0.586) du SHAP xac nhan XGBoost DUNG RAT NHIEU feature svdd_dist (5-6/15 top).
Nguyen nhan: XGBoost hoc LUAT LIEN HOP "svdd_dist cao NHUNG Dst Port quen =>
benign", lam mat tin hieu one-class tren attack chua tung thay. Bang chung:
SSH_Bruteforce SVDD-Only=1.0000 nhung SVDD+XGB=0.0001.
Xem docstring common/xgb_config_v5_mono.py de biet chi tiet co che va danh doi.

--- Docstring goc ---
Train SVDD+XGBoost (augmented) — train RIENG 14 lan, moi lan 1 held-out label.
Tai su dung SVDD checkpoint DA TRAIN SAN o artifacts/svdd_only/checkpoint/ (khong
train lai SVDD — SVDD chi hoc Benign 1 lan duy nhat, K_IN sphere dung chung cho ca
14 fold). SVDD CHI dong vai tro feature encoder: K sphere distance + d_min duoc noi
vao 78 feature goc. XGBoost la diem quyet dinh DUY NHAT — khong co nguong SVDD
rieng, khong OR-Fusion (dung thiet ke notebook goc
UTS_Layer1/SVDD_XGBoost/svdd_xgboost_ids.ipynb).

DUNG CHUNG config XGBoost voi train_xgboost_only.py (common/xgb_config.py) de so
sanh cong bang — chi khac o feature input (78 vs 78+K_IN+1).

Pipeline (moi held-out label):
  1. Doc train_benign_attack.csv/val_binary.csv/test_zeroshot.csv cua label do.
  2. Dung CHUNG 1 scaler (svdd_scaler, fit tren benign-only, tai su dung tu
     artifacts/svdd_only/) cho CA XGBoost input LAN SVDD encoder input — dung dung
     pattern notebook goc (Cell 6+20: scaler.fit_transform tren benign, sau do
     scaler.transform() 1 lan duy nhat, KHONG co scaler rieng cho XGBoost). Ban truoc
     dung 2 scaler khac nhau (mot fit tren train_benign_attack, mot fit tren
     benign-only) — day la loi kien truc khien SVDD-encoder va XGBoost nhin thay 2
     phien ban du lieu da qua bien doi phan phoi khac nhau tren cung 1 tap X_raw,
     khien XGBoost khong hoc duoc cach dung sphere-distance feature (da xac nhan qua
     thuc nghiem: chi them clip [-20,20] KHONG giai quyet duoc — SVDD+XGBoost van
     fail y het cac nhan ma XGBoost-only (khong SVDD) cung fail).
  3. Dung SVDD (da load K_IN sphere) tinh sphere distances (K_IN feature) + d_min (1)
     tren CHINH du lieu da scale o buoc 2 (khong con 2 phien ban rieng biet nua).
  4. Ghep [X_scaled | sphere_dists(K_IN) | d_min] feature, train XGBClassifier.
  5. Tune threshold F1-max tren TRAIN (train_benign_attack.csv), khong phai val —
     vi val_binary.csv la tap DUNG DE BAO CAO known-attack F1/recall/FPR; tune
     threshold tren chinh tap eval se lam ket qua bi thoi phong (data leak kieu
     "threshold biet truoc dap an"). Sau khi train xong CA 14 fold, lay TRUNG
     BINH 14 threshold nay thanh 1 threshold duy nhat, roi dung LAI threshold
     chung do de eval lai toan bo 14 fold (known-attack + zero-shot) — mo phong
     dung mot nguong quyet dinh co dinh nhu khi trien khai thuc te.
  6. Luu checkpoint (model.pkl) vao artifacts/svdd_xgboost/<label>/checkpoint/
     (khong con scaler rieng — dung chung svdd_scaler tu artifacts/svdd_only/).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.xgb_config_v5_mono import (  # noqa: E402
    XGB_PARAMS,
    XGB_SCALE_POS_DIVISOR,
    XGB_MONOTONE_AUGMENTED,
    build_monotone_constraints,
)
from models.svdd import MixtureSVDD  # noqa: E402
from models.clad.util.metrics import best_f1_threshold  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = BASE_DIR / "03_splits_v2"
LABELS_CONFIG_PATH = BASE_DIR / "config" / "labels.json"

SVDD_ONLY_CKPT_DIR = Path(__file__).resolve().parent / "artifacts" / "svdd_only" / "checkpoint"
SVDD_SCALER_PATH = SVDD_ONLY_CKPT_DIR / "scaler.pkl"
SVDD_ELBOW_CACHE_PATH = SVDD_ONLY_CKPT_DIR / "elbow_cache.json"
SVDD_MODEL_DIR = SVDD_ONLY_CKPT_DIR / "svdd_model"

ARTIFACTS_ROOT = Path(__file__).resolve().parent / "artifacts" / "svdd_xgboost_mono"
METRICS_PATH = ARTIFACTS_ROOT / "metrics.json"

SEED = 42
np.random.seed(SEED)

CLIP_VAL = 20.0

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"
XGB_DEVICE = DEVICE


def load_split_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "Label" in df.columns and "Label_Text" in df.columns, f"{path}: thieu cot Label/Label_Text"
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("Label", "Label_Text")]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tra ve dung 6 chi so: f1, recall, precision, accuracy, fpr, n_samples."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(rec),
        "precision": float(pre),
        "accuracy": float(acc),
        "fpr": float(fpr),
        "n_samples": int(len(y_true)),
    }


def train_one_label(label: str, feature_cols: list[str], svdd: MixtureSVDD, svdd_scaler, k_in: int) -> dict:
    label_artifacts_dir = ARTIFACTS_ROOT / label
    images_dir = label_artifacts_dir / "images"
    ckpt_dir = label_artifacts_dir / "checkpoint"
    images_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_path = ckpt_dir / "model.pkl"

    train_df = load_split_csv(SPLITS_DIR / "by_label" / label / "train_benign_attack.csv")
    val_df = load_split_csv(SPLITS_DIR / "by_label" / label / "val_binary.csv")
    test_df = load_split_csv(SPLITS_DIR / "by_label" / label / "test_zeroshot.csv")

    X_train_raw = train_df[feature_cols].values.astype("float32")
    y_train = train_df["Label"].values
    X_val_raw = val_df[feature_cols].values.astype("float32")
    y_val = val_df["Label"].values
    X_test_raw = test_df[feature_cols].values.astype("float32")
    y_test = test_df["Label"].values

    # Dung CHUNG 1 scaler (svdd_scaler, fit tren benign-only tu svdd_only) cho CA
    # XGBoost input LAN SVDD encoder input — dung 100% pattern cua notebook goc
    # UTS_Layer1/SVDD_XGBoost/svdd_xgboost_ids.ipynb Cell 6+20 (scaler.fit_transform
    # tren benign, roi scaler.transform() DUY NHAT MOT LAN cho ca X_xgb_train VA lam
    # input tinh sphere distance — KHONG co "scaler_xgb" rieng biet).
    #
    # Ban truoc cua file nay dung 2 scaler khac nhau (scaler_xgb fit tren
    # train_benign_attack, svdd_scaler fit tren benign-only) cho 2 nhanh feature khac
    # nhau tren CUNG 1 tap X_train_raw — khien SVDD-encoder nhin thay du lieu qua 1
    # phep bien doi phan phoi khac voi XGBoost, dan den XGBoost khong hoc duoc cach
    # dung dung sphere-distance feature (da xac nhan bang thuc nghiem: chi them clip
    # khong giai quyet duoc van de, SVDD+XGBoost van fail y het XGBoost-only tren
    # cung nhung nhan — dau hieu ro rang day la loi kien truc scaler, khong phai loi
    # outlier/scale bien do).
    X_train_xgb_s = np.clip(svdd_scaler.transform(X_train_raw), -CLIP_VAL, CLIP_VAL)
    X_val_xgb_s = np.clip(svdd_scaler.transform(X_val_raw), -CLIP_VAL, CLIP_VAL)
    X_test_xgb_s = np.clip(svdd_scaler.transform(X_test_raw), -CLIP_VAL, CLIP_VAL)

    dists_train = svdd.all_sphere_distances(X_train_xgb_s)
    dists_val = svdd.all_sphere_distances(X_val_xgb_s)
    dists_test = svdd.all_sphere_distances(X_test_xgb_s)

    X_train_aug = np.concatenate([X_train_xgb_s, dists_train, dists_train.min(axis=1, keepdims=True)], axis=1)
    X_val_aug = np.concatenate([X_val_xgb_s, dists_val, dists_val.min(axis=1, keepdims=True)], axis=1)
    X_test_aug = np.concatenate([X_test_xgb_s, dists_test, dists_test.min(axis=1, keepdims=True)], axis=1)

    scale_pos = float((y_train == 0).sum()) / float((y_train == 1).sum())

    # Monotone constraint: thu tu cot la [78 goc | k_in sphere_dist | 1 d_min]
    # (khop chinh xac voi np.concatenate o tren). Ca k_in sphere_dist LAN d_min
    # deu co ngu nghia "cang xa vung benign => cang bat thuong" nen deu rang
    # buoc +1; 78 feature goc giu tu do (0).
    n_base = len(feature_cols)
    mono = build_monotone_constraints(n_base=n_base, n_augmented_monotone=k_in + 1)
    assert mono is None or len(mono) == X_train_aug.shape[1], (
        f"monotone_constraints do dai {len(mono)} != so cot {X_train_aug.shape[1]}"
    )

    if model_path.exists():
        model = joblib.load(model_path)
    else:
        model = xgb.XGBClassifier(
            **XGB_PARAMS,
            scale_pos_weight=scale_pos / XGB_SCALE_POS_DIVISOR,
            monotone_constraints=mono,
            device=XGB_DEVICE,
        )
        t0 = time.time()
        model.fit(X_train_aug, y_train, eval_set=[(X_val_aug, y_val)], verbose=False)
        print(f"    Training time: {time.time()-t0:.1f}s")
        joblib.dump(model, model_path)

    proba_train = model.predict_proba(X_train_aug)[:, 1]
    proba_val = model.predict_proba(X_val_aug)[:, 1]
    proba_test = model.predict_proba(X_test_aug)[:, 1]

    # Tune threshold theo F1-max tren TRAIN (khong phai val) — vi val_binary la
    # tap dung de BAO CAO known-attack F1/recall/FPR, tune tren chinh no se lam
    # threshold "biet truoc" tap eval, thoi phong ket qua. Threshold nay se duoc
    # gop trung binh qua 14 fold o main() roi dung LAI 1 threshold chung de
    # danh gia lai toan bo (xem main()).
    threshold, f1_train_opt, prec_train_opt, rec_train_opt = best_f1_threshold(y_train, proba_train)
    print(f"    [Train] best-F1 threshold={threshold:.4f}  F1={f1_train_opt:.4f}  "
          f"P={prec_train_opt:.4f}  R={rec_train_opt:.4f}")

    # SHAP top-15 (danh dau ro feature nao la SVDD-augmented)
    shap_path = images_dir / "shap_bar_top15.png"
    if not shap_path.exists():
        try:
            import shap
            augmented_feature_names = feature_cols + [f"svdd_dist_{k}" for k in range(k_in)] + ["svdd_dmin"]
            shap_idx = np.random.choice(len(X_val_aug), min(2000, len(X_val_aug)), replace=False)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val_aug[shap_idx])
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top15_idx = np.argsort(mean_abs_shap)[::-1][:15]
            top15_features = [augmented_feature_names[i] for i in top15_idx]
            colors = ["#e74c3c" if "svdd_" in f else "#3498db" for f in top15_features]

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(range(len(top15_features)), mean_abs_shap[top15_idx][::-1], color=colors[::-1])
            ax.set_yticks(range(len(top15_features)))
            ax.set_yticklabels(top15_features[::-1], fontsize=9)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"SVDD+XGBoost SHAP Top 15 — held-out={label}\nRed = SVDD feature, Blue = original")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(shap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except ImportError:
            pass

    return {
        "threshold_from_train": threshold,
        "proba_val": proba_val,
        "y_val": y_val,
        "proba_test": proba_test,
        "y_test": y_test,
    }


def main() -> None:
    with open(LABELS_CONFIG_PATH, "r", encoding="utf-8") as f:
        labels_cfg = json.load(f)
    attack_labels = labels_cfg["attack_labels"]

    assert SVDD_SCALER_PATH.exists() and SVDD_ELBOW_CACHE_PATH.exists(), (
        f"Khong tim thay SVDD checkpoint tai {SVDD_ONLY_CKPT_DIR} — chay train_svdd_only.py truoc."
    )
    print(f"Dang load SVDD checkpoint tu {SVDD_ONLY_CKPT_DIR} ...")
    svdd_scaler = joblib.load(SVDD_SCALER_PATH)

    with open(SVDD_ELBOW_CACHE_PATH, "r", encoding="utf-8") as f:
        elbow_cache = json.load(f)
    k_chosen = elbow_cache["k_chosen"]

    sample_df = load_split_csv(SPLITS_DIR / "by_label" / attack_labels[0] / "train_benign_attack.csv")
    feature_cols = get_feature_cols(sample_df)

    # k_in (so sphere sau khi tach cluster lon nhat) suy tu so file encoder_*.pt da luu
    encoder_files = sorted(SVDD_MODEL_DIR.glob("encoder_*.pt"))
    k_in = len(encoder_files)
    assert k_in > 0, f"Khong tim thay encoder nao trong {SVDD_MODEL_DIR}"

    svdd = MixtureSVDD(K=k_in, input_dim=len(feature_cols), embed_dim=64, device=DEVICE)
    svdd.load(str(SVDD_MODEL_DIR))
    print(f"XGBoost device: {XGB_DEVICE}  |  {len(feature_cols)} feature goc + {k_in} sphere-dist + 1 d_min")

    results: dict = {"xgb_params": XGB_PARAMS, "k_chosen": k_chosen, "k_in": k_in,
                      "monotone_augmented": XGB_MONOTONE_AUGMENTED,
                      "monotone_n_constrained": (k_in + 1) if XGB_MONOTONE_AUGMENTED else 0,
                      "scale_pos_divisor": XGB_SCALE_POS_DIVISOR,
                      "zero_shot_per_label": {}, "known_attack_per_label": {}}
    print(f"Monotone constraint: {'BAT' if XGB_MONOTONE_AUGMENTED else 'TAT'} "
          f"({k_in + 1} feature SVDD bi rang buoc don dieu tang)")

    # Phase 1: train tat ca 14 fold, chi thu threshold+proba tho, chua eval.
    raw_results: dict[str, dict] = {}
    for label in attack_labels:
        print(f"\n=== held-out = {label} ===")
        raw_results[label] = train_one_label(label, feature_cols, svdd, svdd_scaler, k_in)

    # Phase 2: tinh threshold trung binh tu 14 fold, danh gia lai CA 14 fold voi
    # CUNG 1 threshold nay (mo phong deployment thuc te chi co 1 nguong co dinh).
    threshold_per_label = {label: float(r["threshold_from_train"]) for label, r in raw_results.items()}
    threshold_mean = float(np.mean(list(threshold_per_label.values())))
    print(f"\n=== Threshold trung binh (tu 14 fold, tune tren train) = {threshold_mean:.4f} ===")
    print(f"    Threshold tung fold: { {k: round(v, 4) for k, v in threshold_per_label.items()} }")

    for label in attack_labels:
        r = raw_results[label]
        pred_val = (r["proba_val"] >= threshold_mean).astype(int)
        pred_test = (r["proba_test"] >= threshold_mean).astype(int)

        known_metrics = evaluate(r["y_val"], pred_val)
        zero_shot_metrics = evaluate(r["y_test"], pred_test)
        zero_shot_metrics["zero_acc"] = zero_shot_metrics["accuracy"]

        results["zero_shot_per_label"][label] = zero_shot_metrics
        results["known_attack_per_label"][label] = known_metrics
        print(f"  {label}: zero-shot recall={zero_shot_metrics['recall']:.4f} f1={zero_shot_metrics['f1']:.4f}  |  "
              f"known-attack recall={known_metrics['recall']:.4f} f1={known_metrics['f1']:.4f} "
              f"fpr={known_metrics['fpr']:.4f}")

    results["threshold_per_label_from_train"] = threshold_per_label
    results["threshold_mean_used_for_eval"] = threshold_mean

    def macro_avg(per_label: dict, key: str) -> float:
        vals = [m[key] for m in per_label.values() if m[key] is not None]
        return float(np.mean(vals)) if vals else None

    results["summary"] = {
        "zero_shot_macro_avg": {
            "f1": macro_avg(results["zero_shot_per_label"], "f1"),
            "recall": macro_avg(results["zero_shot_per_label"], "recall"),
            "precision": macro_avg(results["zero_shot_per_label"], "precision"),
            "zero_acc": macro_avg(results["zero_shot_per_label"], "zero_acc"),
            "fpr": macro_avg(results["zero_shot_per_label"], "fpr"),
        },
        "known_attack_macro_avg": {
            "f1": macro_avg(results["known_attack_per_label"], "f1"),
            "recall": macro_avg(results["known_attack_per_label"], "recall"),
            "precision": macro_avg(results["known_attack_per_label"], "precision"),
            "accuracy": macro_avg(results["known_attack_per_label"], "accuracy"),
            "fpr": macro_avg(results["known_attack_per_label"], "fpr"),
        },
    }
    print("\n=== SUMMARY (macro avg qua 14 held-out fold) ===")
    zs = results["summary"]["zero_shot_macro_avg"]
    ka = results["summary"]["known_attack_macro_avg"]
    print(f"  Zero-shot   : f1={zs['f1']:.4f}  recall={zs['recall']:.4f}  precision={zs['precision']:.4f}  "
          f"zero_acc={zs['zero_acc']:.4f}  fpr={zs['fpr']:.4f}")
    print(f"  Known-attack: f1={ka['f1']:.4f}  recall={ka['recall']:.4f}  precision={ka['precision']:.4f}  "
          f"accuracy={ka['accuracy']:.4f}  fpr={ka['fpr']:.4f}")

    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDa ghi metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
