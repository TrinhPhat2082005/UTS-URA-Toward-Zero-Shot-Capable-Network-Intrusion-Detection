"""
Train SVDD-Only baseline — MOT LAN DUY NHAT tren 03_splits/train_benign_only.csv
(khong lap lai theo tung held-out label, vi SVDD one-class chi hoc Benign).

Kien truc + pipeline copy nguyen tu UTS_Layer1/SVDD_XGBoost/svdd_xgboost_ids.ipynb
(Cell 1-9), CHI thay doi 1 diem: K_CHOSEN duoc chon TU DONG qua
argmax(silhouette_score) trong so cac K > MIN_K_ALLOWED, thay vi hardcode=8 nhin
bieu do thu cong (yeu cau 2026-08-27 cua user — xem plan.md). Gioi han
MIN_K_ALLOWED duoc them 2026-09-09 vi tren data v2 (560K dong benign), silhouette
dao dong nhieu va argmax khong gioi han roi vao K=2 (qua tho, chi hon kem ~0.003
so voi K=10) — xem RESULTS.md muc dieu tra K-Means v2. Cac buoc khac giu nguyen 100%:
  - RobustScaler fit tren benign train, clip [-20, 20].
  - K-Means (K=K_CHOSEN tu dong) tren toan bo benign.
  - Tach RIENG cluster LON NHAT thanh K_SPLIT=3 sub-cluster.
  - Loai cluster/sub-cluster < MIN_SIZE=2048 samples.
  - Train 1 sphere MixtureSVDD rieng cho moi cluster con lai (K_IN spheres).

Pipeline:
  1. Doc train_benign_only.csv, RobustScaler, clip.
  2. select_optimal_k(): thu K trong [2..10] tren subsample 100K, chon K co
     silhouette_score cao nhat — luu elbow+silhouette curve (danh dau K duoc chon).
  3. K-Means K=K_CHOSEN tren TOAN BO benign train -> tach cluster lon nhat
     thanh 3 sub-cluster -> loai cluster nho -> K_IN cluster cuoi cung.
  4. Train MixtureSVDD (K_IN sphere doc lap, nu=0.01, lr=1e-3, epochs=50).
  5. Threshold + score distribution tren val_binary.csv cua 1 label dai dien.
  6. Eval tren CA 14 test_zeroshot.csv, ghi metrics.json.

KHONG train XGBoost o day — file nay CHI danh gia SVDD nhu 1 detector doc lap
(baseline "SVDD-Only" trong 6-baseline comparison).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, f1_score, silhouette_score
from sklearn.preprocessing import RobustScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.svdd import MixtureSVDD  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = BASE_DIR / "03_splits_v2"
LABELS_CONFIG_PATH = BASE_DIR / "config" / "labels.json"

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts" / "svdd_only"
IMAGES_DIR = ARTIFACTS_DIR / "images"
CKPT_DIR = ARTIFACTS_DIR / "checkpoint"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

for d in (IMAGES_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SCALER_PATH = CKPT_DIR / "scaler.pkl"
KMEANS_ASSIGN_PATH = CKPT_DIR / "kmeans_assign.pkl"
KMEANS_SUB_PATH = CKPT_DIR / "kmeans_sub.pkl"
SVDD_MODEL_DIR = CKPT_DIR / "svdd_model"

SEED = 42
np.random.seed(SEED)

CLIP_VAL = 20.0
MAX_ELBOW = 100_000
K_RANGE = range(2, 11)
# Silhouette tren v2 (560K dong benign) dao dong gan nhu giong het v1 (K=2..9) nhung
# chenh lech K=2 vs K=10 chi ~0.003 — argmax nhay tu K=10 (v1, ~100K dong) xuong K=2
# (v2) chi vi nhieu ngau nhien (subsample 100K de tinh silhouette), khong phai vi
# K=2 phan anh dung cau truc cluster benign thuc su. De tranh chon nham K qua nho
# (2 cluster qua tho cho 560K dong benign da dang), gioi han K > MIN_K_ALLOWED —
# chon K co silhouette cao nhat trong cac K con lai thoa dieu kien (yeu cau nguoi
# dung 2026-09-09, de dang chinh lai neu can doi gioi han).
MIN_K_ALLOWED = 6
K_SPLIT = 3
MIN_SIZE = 2048
NU = 0.01
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 2048
FPR_BUDGET = 0.05

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"


def load_split_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "Label" in df.columns and "Label_Text" in df.columns, f"{path}: thieu cot Label/Label_Text"
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("Label", "Label_Text")]


def select_optimal_k(X_elbow: np.ndarray) -> tuple[int, list[float], list[float]]:
    """Thu K trong K_RANGE, chon K co silhouette_score cao nhat trong so cac K
    THOA K > MIN_K_ALLOWED (argmax co gioi han duoi).
    Ly do gioi han: tren du lieu benign thuc te (560K dong, da dang hon ban cap
    nho 100K truoc day), duong cong silhouette dao dong nhieu va khong on dinh —
    argmax khong gioi han co the roi vao K rat nho (vd K=2) chi vi nhieu ngau
    nhien (subsample de tinh silhouette), chu khong phan anh dung cau truc
    cluster benign. MIN_K_ALLOWED tranh chon K qua tho."""
    inertias, sil_scores = [], []
    print("Dang chay K-Means elbow (auto-select K qua silhouette)...")
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=5, max_iter=200)
        labels = km.fit_predict(X_elbow)
        inertias.append(float(km.inertia_))
        sil_idx = np.random.choice(len(X_elbow), min(10_000, len(X_elbow)), replace=False)
        sil = float(silhouette_score(X_elbow[sil_idx], labels[sil_idx]))
        sil_scores.append(sil)
        print(f"  K={k}  inertia={km.inertia_:.2f}  silhouette={sil:.4f}")

    k_list = list(K_RANGE)
    allowed_idx = [i for i, k in enumerate(k_list) if k > MIN_K_ALLOWED]
    assert allowed_idx, f"Khong co K nao > MIN_K_ALLOWED={MIN_K_ALLOWED} trong K_RANGE={K_RANGE}"
    best_i = max(allowed_idx, key=lambda i: sil_scores[i])
    k_chosen = k_list[best_i]
    print(f"K_CHOSEN (argmax silhouette, K > {MIN_K_ALLOWED}) = {k_chosen}")
    return k_chosen, inertias, sil_scores


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


def main() -> None:
    with open(LABELS_CONFIG_PATH, "r", encoding="utf-8") as f:
        labels_cfg = json.load(f)
    attack_labels = labels_cfg["attack_labels"]

    print("Dang doc train_benign_only.csv ...")
    train_df = load_split_csv(SPLITS_DIR / "train_benign_only.csv")
    feature_cols = get_feature_cols(train_df)
    print(f"  {len(train_df):,} dong Benign, {len(feature_cols)} feature.")

    X_train_raw = train_df[feature_cols].values.astype("float32")

    if SCALER_PATH.exists():
        print(f"[Checkpoint] Found {SCALER_PATH} - loading scaler.")
        scaler = joblib.load(SCALER_PATH)
    else:
        scaler = RobustScaler()
        scaler.fit(X_train_raw)
        joblib.dump(scaler, SCALER_PATH)
        print(f"RobustScaler fit tren benign train - saved to {SCALER_PATH}")

    X_train_s = np.clip(scaler.transform(X_train_raw), -CLIP_VAL, CLIP_VAL)

    # ---- Chon K tu dong ----
    elbow_cache = CKPT_DIR / "elbow_cache.json"
    idx = np.random.choice(len(X_train_s), min(MAX_ELBOW, len(X_train_s)), replace=False)
    X_elbow = X_train_s[idx]

    if elbow_cache.exists():
        with open(elbow_cache, "r", encoding="utf-8") as f:
            cache = json.load(f)
        k_chosen, inertias, sil_scores = cache["k_chosen"], cache["inertias"], cache["sil_scores"]
        print(f"[Checkpoint] Found {elbow_cache} - K_CHOSEN={k_chosen}, skip elbow.")
    else:
        k_chosen, inertias, sil_scores = select_optimal_k(X_elbow)
        with open(elbow_cache, "w", encoding="utf-8") as f:
            json.dump({"k_chosen": k_chosen, "inertias": inertias, "sil_scores": sil_scores}, f, indent=2)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(list(K_RANGE), inertias, "bo-", linewidth=2, markersize=8)
        axes[0].axvline(k_chosen, color="green", linestyle="--", label=f"K_chosen={k_chosen}")
        axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia (WCSS)")
        axes[0].set_title("Elbow Curve - Inertia"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(list(K_RANGE), sil_scores, "rs-", linewidth=2, markersize=8)
        axes[1].axvline(k_chosen, color="green", linestyle="--", label=f"K_chosen={k_chosen}")
        axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
        axes[1].set_title("Silhouette Score (auto argmax)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.suptitle("K-Means Analysis on Benign Traffic (zeroshot_benchmark)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "kmeans_elbow.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Da luu {IMAGES_DIR / 'kmeans_elbow.png'}")

    # ---- K-Means K=k_chosen tren toan bo benign, tach cluster lon nhat ----
    if KMEANS_ASSIGN_PATH.exists() and KMEANS_SUB_PATH.exists():
        print(f"[Checkpoint] Found {KMEANS_ASSIGN_PATH} - loading K-Means.")
        km_assign = joblib.load(KMEANS_ASSIGN_PATH)
        km_sub = joblib.load(KMEANS_SUB_PATH)
        cluster_assign = km_assign.predict(X_train_s)
        raw_clusters = {k: X_train_s[cluster_assign == k] for k in range(k_chosen)}
        largest_k = max(raw_clusters, key=lambda k: len(raw_clusters[k]))
        X_largest = raw_clusters[largest_k]
        sub_assign = km_sub.predict(X_largest)
        sub_clusters = {j: X_largest[sub_assign == j] for j in range(K_SPLIT)}
    else:
        print(f"Dang chay K-Means K={k_chosen} tren toan bo benign...")
        km_assign = KMeans(n_clusters=k_chosen, random_state=SEED, n_init=10)
        cluster_assign = km_assign.fit_predict(X_train_s)
        raw_clusters = {}
        for k in range(k_chosen):
            X_k = X_train_s[cluster_assign == k]
            raw_clusters[k] = X_k
            print(f"  Cluster {k}: {len(X_k):,} samples")

        largest_k = max(raw_clusters, key=lambda k: len(raw_clusters[k]))
        X_largest = raw_clusters[largest_k]
        print(f"Cluster lon nhat: #{largest_k} ({len(X_largest):,} samples) -> tach thanh {K_SPLIT} sub-cluster")

        km_sub = KMeans(n_clusters=K_SPLIT, random_state=SEED, n_init=10)
        sub_assign = km_sub.fit_predict(X_largest)
        sub_clusters = {j: X_largest[sub_assign == j] for j in range(K_SPLIT)}
        for j, X_j in sub_clusters.items():
            print(f"    Sub-cluster {largest_k}.{j}: {len(X_j):,} samples")

        joblib.dump(km_assign, KMEANS_ASSIGN_PATH)
        joblib.dump(km_sub, KMEANS_SUB_PATH)

    X_clusters = []
    print(f"Tong hop - loai cluster < {MIN_SIZE} samples:")
    for k, X_k in raw_clusters.items():
        if k == largest_k:
            for j, X_j in sub_clusters.items():
                if len(X_j) >= MIN_SIZE:
                    X_clusters.append(X_j)
                    print(f"  OK Sub {k}.{j}: {len(X_j):,} -> giu")
                else:
                    print(f"  X  Sub {k}.{j}: {len(X_j):,} -> loai")
        else:
            if len(X_k) >= MIN_SIZE:
                X_clusters.append(X_k)
                print(f"  OK Cluster {k}: {len(X_k):,} -> giu")
            else:
                print(f"  X  Cluster {k}: {len(X_k):,} -> loai")

    k_in = len(X_clusters)
    print(f"Tong so sphere SVDD se train: {k_in}")

    # ---- Train MixtureSVDD ----
    svdd = MixtureSVDD(
        K=k_in, input_dim=len(feature_cols), embed_dim=64,
        nu=NU, lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
    )

    svdd_ckpt_complete = (
        (SVDD_MODEL_DIR / "centers.npy").exists()
        and (SVDD_MODEL_DIR / "radii.npy").exists()
        and all((SVDD_MODEL_DIR / f"encoder_{k}.pt").exists() for k in range(k_in))
    )
    if svdd_ckpt_complete:
        print(f"[Checkpoint] Found SVDD checkpoint tai {SVDD_MODEL_DIR} - loading.")
        svdd.load(str(SVDD_MODEL_DIR))
    else:
        t0 = time.time()
        svdd.fit(X_clusters)
        print(f"Total SVDD training time: {time.time()-t0:.1f}s")
        svdd.save(str(SVDD_MODEL_DIR))

    # ---- Threshold + distribution tren val_binary.csv cua 1 label dai dien ----
    threshold_label = attack_labels[0]
    print(f"\nDang chon threshold tren val_binary.csv (held-out={threshold_label}) ...")
    val_ref_df = load_split_csv(SPLITS_DIR / "by_label" / threshold_label / "val_binary.csv")
    X_val_ref_raw = val_ref_df[feature_cols].values.astype("float32")
    X_val_ref_s = np.clip(scaler.transform(X_val_ref_raw), -CLIP_VAL, CLIP_VAL)
    y_val_ref = val_ref_df["Label"].values

    scores_val_ref = svdd.decision_scores(X_val_ref_s)
    from sklearn.metrics import precision_recall_curve
    precision_c, recall_c, thresholds_c = precision_recall_curve(y_val_ref, scores_val_ref)
    f1_curve = 2 * precision_c * recall_c / (precision_c + recall_c + 1e-9)
    best_idx = f1_curve[:-1].argmax() if len(f1_curve) > 1 else 0
    svdd_threshold = float(thresholds_c[best_idx]) if len(thresholds_c) > 0 else float(np.median(scores_val_ref))
    print(f"  SVDD threshold (F1-max tren val): {svdd_threshold:.6f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    clip_p = np.percentile(scores_val_ref, 99)
    sc = np.clip(scores_val_ref, 0, clip_p)
    ax.hist(sc[y_val_ref == 0], bins=80, alpha=0.6, color="green", label="Benign", density=True)
    ax.hist(sc[y_val_ref == 1], bins=80, alpha=0.6, color="red", label="Attack", density=True)
    ax.axvline(min(svdd_threshold, clip_p), color="black", linestyle="--", linewidth=2,
               label=f"Threshold={svdd_threshold:.4f}")
    ax.set_title(f"SVDD d_min Distribution — val_binary ({threshold_label} held-out)")
    ax.set_xlabel("d_min (distance to nearest center)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "svdd_score_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Da luu {IMAGES_DIR / 'svdd_score_dist.png'}")

    # ---- Eval ZERO-SHOT tren CA 14 test_zeroshot.csv ----
    print("\nDang eval ZERO-SHOT tren 14 test_zeroshot.csv (chi attack bi giu lai, khong Benign) ...")
    results: dict = {
        "k_chosen": k_chosen, "k_in_after_split": k_in,
        "threshold_chosen_on_label": threshold_label, "svdd_threshold": svdd_threshold,
        "zero_shot_per_label": {}, "known_attack_per_label": {},
    }

    for label in attack_labels:
        test_path = SPLITS_DIR / "by_label" / label / "test_zeroshot.csv"
        test_df = load_split_csv(test_path)
        X_test_raw = test_df[feature_cols].values.astype("float32")
        X_test_s = np.clip(scaler.transform(X_test_raw), -CLIP_VAL, CLIP_VAL)
        y_test = test_df["Label"].values

        scores_test = svdd.decision_scores(X_test_s)
        pred_test = (scores_test > svdd_threshold).astype(int)

        metrics = evaluate(y_test, pred_test)
        metrics["zero_acc"] = metrics["accuracy"]
        results["zero_shot_per_label"][label] = metrics
        print(f"  {label:30s} recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}  n={metrics['n_samples']:,}")

    # ---- Known-attack eval: chay tren val_binary.csv (Benign + 13 attack KNOWN) cua
    # ca 14 held-out fold, lay trung binh macro. Khac zero-shot o cho: co Benign that
    # trong tap nay nen FPR/Precision co y nghia thong ke that.
    print("\nDang eval KNOWN-ATTACK tren 14 val_binary.csv (Benign + 13 attack known, moi held-out fold) ...")
    for label in attack_labels:
        val_path = SPLITS_DIR / "by_label" / label / "val_binary.csv"
        val_df = load_split_csv(val_path)
        X_val_raw = val_df[feature_cols].values.astype("float32")
        X_val_s = np.clip(scaler.transform(X_val_raw), -CLIP_VAL, CLIP_VAL)
        y_val = val_df["Label"].values

        scores_val = svdd.decision_scores(X_val_s)
        pred_val = (scores_val > svdd_threshold).astype(int)

        metrics = evaluate(y_val, pred_val)
        results["known_attack_per_label"][label] = metrics
        print(f"  held-out={label:30s} recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}  "
              f"fpr={metrics['fpr']:.4f}  n={metrics['n_samples']:,}")

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

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDa ghi metrics: {METRICS_PATH}")
    print(f"Artifacts: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
