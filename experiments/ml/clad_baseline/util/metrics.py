"""
Evaluation metrics for binary anomaly detection.
Convention: anomaly_score HIGH = attack (same as sklearn's precision_recall_curve).
CLAD cosine similarity is LOW=attack, so callers should negate before passing here.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score,
    classification_report, confusion_matrix,
)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray):
    """PR-curve F1 optimisation on a given split (use val, not test)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    idx = np.argmax(f1)
    thr = thresholds[min(idx, len(thresholds) - 1)]
    return thr, f1[idx], precision[idx], recall[idx]


def fpr_at_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float = 0.95):
    """False-positive rate when recall ≥ target_recall."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(tpr, target_recall)
    return float(fpr[min(idx, len(fpr) - 1)])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float, label: str = ""):
    """Full evaluation given anomaly scores and a fixed threshold."""
    y_pred = (scores >= threshold).astype(int)

    auc_roc = roc_auc_score(y_true, scores)
    auc_pr  = average_precision_score(y_true, scores)
    fpr95   = fpr_at_recall(y_true, scores, 0.95)
    f1      = f1_score(y_true, y_pred, zero_division=0)

    header = f"\n{'='*55}\n  {label}\n{'='*55}"
    print(header)
    print(classification_report(y_true, y_pred,
                                target_names=['Benign', 'Attack'], digits=4))
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = (cm[0, 0], 0, 0, 0)
    print(f"  AUC-ROC : {auc_roc:.4f}")
    print(f"  AUC-PR  : {auc_pr:.4f}")
    print(f"  FPR@95R : {fpr95:.4f}")
    print(f"  F1      : {f1:.4f}")
    print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

    return {
        'auc_roc': auc_roc, 'auc_pr': auc_pr,
        'fpr95': fpr95, 'f1': f1,
        'threshold': threshold,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }
