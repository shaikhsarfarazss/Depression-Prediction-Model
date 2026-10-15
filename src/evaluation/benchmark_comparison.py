import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVC
from .metrics import calculate_clinical_metrics
from ..models.hybrid_ensemble import HybridCNNSVMKNNEnsemble

def run_benchmark_comparison(
    ensemble: HybridCNNSVMKNNEnsemble,
    X_test_3d: np.ndarray,
    X_test_2d: np.ndarray,
    y_test: np.ndarray,
    X_train_2d: np.ndarray = None,
    y_train: np.ndarray = None
) -> pd.DataFrame:
    results = []

    # 1. Proposed Hybrid (CNN-SVM-KNN)
    y_pred_hybrid = ensemble.predict(X_test_3d)
    y_prob_hybrid = ensemble.predict_proba(X_test_3d)
    m_hybrid = calculate_clinical_metrics(y_test, y_pred_hybrid, y_prob_hybrid)
    results.append({
        "Model": "Proposed Hybrid (CNN-SVM-KNN)",
        "Type": "Full Hybrid Ensemble",
        "Accuracy (%)": m_hybrid["overall_accuracy"],
        "Sensitivity (%)": m_hybrid["macro_sensitivity"],
        "Specificity (%)": m_hybrid["macro_specificity"],
        "Precision (%)": m_hybrid["macro_precision"],
        "F1-Score (%)": m_hybrid["macro_f1"],
        "ROC-AUC (%)": m_hybrid["roc_auc"]
    })

    # 2. CNN Alone
    ensemble.cnn.eval()
    with torch.no_grad():
        t_x = torch.tensor(X_test_3d, dtype=torch.float32).to(ensemble.device)
        cnn_probs = ensemble.cnn.predict_proba(t_x).cpu().numpy()
        cnn_preds = np.argmax(cnn_probs, axis=1)
    m_cnn = calculate_clinical_metrics(y_test, cnn_preds, cnn_probs)
    results.append({
        "Model": "CNN Alone",
        "Type": "Deep Learning Baseline",
        "Accuracy (%)": m_cnn["overall_accuracy"],
        "Sensitivity (%)": m_cnn["macro_sensitivity"],
        "Specificity (%)": m_cnn["macro_specificity"],
        "Precision (%)": m_cnn["macro_precision"],
        "F1-Score (%)": m_cnn["macro_f1"],
        "ROC-AUC (%)": m_cnn["roc_auc"]
    })

    # 3. CNN + SVM Pairwise
    deep_test = ensemble.extract_deep_features(X_test_3d)
    svm_probs = ensemble.svm.predict_proba(deep_test)
    svm_preds = np.argmax(svm_probs, axis=1)
    m_svm = calculate_clinical_metrics(y_test, svm_preds, svm_probs)
    results.append({
        "Model": "CNN + SVM",
        "Type": "Pairwise Hybrid",
        "Accuracy (%)": m_svm["overall_accuracy"],
        "Sensitivity (%)": m_svm["macro_sensitivity"],
        "Specificity (%)": m_svm["macro_specificity"],
        "Precision (%)": m_svm["macro_precision"],
        "F1-Score (%)": m_svm["macro_f1"],
        "ROC-AUC (%)": m_svm["roc_auc"]
    })

    # 4. CNN + KNN Pairwise
    knn_probs = ensemble.knn.predict_proba(deep_test)
    knn_preds = np.argmax(knn_probs, axis=1)
    m_knn = calculate_clinical_metrics(y_test, knn_preds, knn_probs)
    results.append({
        "Model": "CNN + KNN",
        "Type": "Pairwise Hybrid",
        "Accuracy (%)": m_knn["overall_accuracy"],
        "Sensitivity (%)": m_knn["macro_sensitivity"],
        "Specificity (%)": m_knn["macro_specificity"],
        "Precision (%)": m_knn["macro_precision"],
        "F1-Score (%)": m_knn["macro_f1"],
        "ROC-AUC (%)": m_knn["roc_auc"]
    })

    # 5. Classical SVM Alone
    if X_train_2d is not None and y_train is not None:
        raw_svm = SVC(probability=True, kernel="rbf", random_state=42)
        raw_svm.fit(X_train_2d, y_train)
        r_svm_probs = raw_svm.predict_proba(X_test_2d)
        r_svm_preds = np.argmax(r_svm_probs, axis=1)
        m_r_svm = calculate_clinical_metrics(y_test, r_svm_preds, r_svm_probs)
        results.append({
            "Model": "Classical SVM Alone",
            "Type": "Shallow Machine Learning",
            "Accuracy (%)": m_r_svm["overall_accuracy"],
            "Sensitivity (%)": m_r_svm["macro_sensitivity"],
            "Specificity (%)": m_r_svm["macro_specificity"],
            "Precision (%)": m_r_svm["macro_precision"],
            "F1-Score (%)": m_r_svm["macro_f1"],
            "ROC-AUC (%)": m_r_svm["roc_auc"]
        })

    return pd.DataFrame(results)
