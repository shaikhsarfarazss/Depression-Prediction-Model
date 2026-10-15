import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
from typing import Dict, Any

CLASS_NAMES = ["Normal", "Alzheimer's Disease", "Depression"]

def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray = None
) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    n_classes = len(CLASS_NAMES)
    specificity_per_class = []
    ovr_accuracy_per_class = []
    total_samples = max(1, len(y_true))

    for i in range(n_classes):
        mask_true_pos = (y_true == i)
        mask_true_neg = (y_true != i)
        tp = np.sum((y_pred == i) & mask_true_pos)
        tn = np.sum((y_pred != i) & mask_true_neg)
        fp = np.sum((y_pred == i) & mask_true_neg)

        spec = tn / (tn + fp + 1e-8)
        ovr_acc = (tp + tn) / float(total_samples)
        specificity_per_class.append(float(spec))
        ovr_accuracy_per_class.append(float(ovr_acc))

    macro_specificity = float(np.mean(specificity_per_class))

    roc_auc = None
    if y_probs is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs, multi_class="ovr"))
        except Exception:
            roc_auc = None

    per_class_metrics = {}
    for i, cname in enumerate(CLASS_NAMES):
        per_class_metrics[cname] = {
            "accuracy": round(ovr_accuracy_per_class[i] * 100, 2),
            "sensitivity": round(float(recall[i]) * 100, 2),
            "specificity": round(float(specificity_per_class[i]) * 100, 2),
            "precision": round(float(precision[i]) * 100, 2),
            "f1_score": round(float(f1[i]) * 100, 2),
            "support": int(support[i])
        }

    return {
        "overall_accuracy": round(acc * 100, 2),
        "macro_sensitivity": round(macro_r * 100, 2),
        "macro_specificity": round(macro_specificity * 100, 2),
        "macro_precision": round(macro_p * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "roc_auc": round(roc_auc * 100, 2) if roc_auc is not None else 0.0,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_metrics
    }
