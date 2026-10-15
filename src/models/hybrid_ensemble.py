import os
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LogisticRegression
from .cnn_model import EEGDeepCNN
from .svm_classifier import SVMClassifier
from .knn_classifier import KNNClassifier
from ..utils.logger import get_logger

logger = get_logger("HybridEnsemble")

CLASS_NAMES = ["Normal", "Alzheimer's Disease", "Depression"]

class HybridCNNSVMKNNEnsemble:
    def __init__(
        self,
        cnn_model: EEGDeepCNN,
        svm_model: Optional[SVMClassifier] = None,
        knn_model: Optional[KNNClassifier] = None,
        fusion_strategy: str = "weighted_voting",
        weights: Optional[Dict[str, float]] = None,
        device: str = "cpu"
    ):
        self.cnn = cnn_model.to(device)
        self.svm = svm_model if svm_model is not None else SVMClassifier()
        self.knn = knn_model if knn_model is not None else KNNClassifier()
        self.fusion_strategy = fusion_strategy
        self.weights = weights or {"cnn": 0.40, "svm": 0.40, "knn": 0.20}
        self.device = device
        self.meta_learner: Optional[LogisticRegression] = None

    def fit_shallow(
        self,
        X_deep: np.ndarray,
        y: np.ndarray,
        X_val_deep: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        p_cnn_val: Optional[np.ndarray] = None,
        train_meta_learner: bool = True
    ):
        logger.info(f"Fitting SVM on {X_deep.shape[0]} deep feature samples...")
        self.svm.fit(X_deep, y)

        logger.info(f"Fitting KNN on {X_deep.shape[0]} deep feature samples...")
        self.knn.fit(X_deep, y)

        if train_meta_learner and self.fusion_strategy == "stacking":
            # Fit stacking meta-learner on validation set to avoid KNN zero-distance memorization
            eval_deep = X_val_deep if X_val_deep is not None else X_deep
            eval_y = y_val if y_val is not None else y
            p_svm = self.svm.predict_proba(eval_deep)
            p_knn = self.knn.predict_proba(eval_deep)
            if p_cnn_val is not None:
                meta_X = np.hstack([p_cnn_val, p_svm, p_knn])
            else:
                meta_X = np.hstack([p_svm, p_knn])
            self.meta_learner = LogisticRegression(max_iter=500, random_state=42)
            self.meta_learner.fit(meta_X, eval_y)
            logger.info("Fitted Stacking Meta-Learner on held-out validation predictions.")

    def extract_deep_features(self, X_3d: np.ndarray) -> np.ndarray:
        self.cnn.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_3d, dtype=torch.float32).to(self.device)
            deep_feats = self.cnn.extract_deep_features(tensor_x)
            return deep_feats.cpu().numpy()

    def predict_detailed(self, X_3d: np.ndarray) -> List[Dict[str, Any]]:
        self.cnn.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_3d, dtype=torch.float32).to(self.device)
            # Single forward pass returns both logits and 64-D deep features
            logits, deep_feats_t = self.cnn(tensor_x)
            cnn_probs = F.softmax(logits, dim=1).cpu().numpy()
            deep_feats = deep_feats_t.cpu().numpy()

        svm_probs = self.svm.predict_proba(deep_feats)
        knn_probs = self.knn.predict_proba(deep_feats)

        batch_results = []
        n_samples = X_3d.shape[0]

        for i in range(n_samples):
            p_cnn = cnn_probs[i]
            p_svm = svm_probs[i]
            p_knn = knn_probs[i]

            if self.fusion_strategy == "stacking" and self.meta_learner is not None:
                expected_features = self.meta_learner.n_features_in_
                if expected_features == 9:
                    meta_in = np.hstack([p_cnn, p_svm, p_knn]).reshape(1, -1)
                else:
                    meta_in = np.hstack([p_svm, p_knn]).reshape(1, -1)
                final_probs = self.meta_learner.predict_proba(meta_in)[0]
            elif self.fusion_strategy == "soft_voting":
                final_probs = (p_cnn + p_svm + p_knn) / 3.0
            else:
                w = self.weights
                final_probs = (
                    w.get("cnn", 0.4) * p_cnn +
                    w.get("svm", 0.4) * p_svm +
                    w.get("knn", 0.2) * p_knn
                )
                final_probs /= np.sum(final_probs)

            pred_class_idx = int(np.argmax(final_probs))
            confidence = float(final_probs[pred_class_idx]) * 100.0

            batch_results.append({
                "predicted_index": pred_class_idx,
                "predicted_label": CLASS_NAMES[pred_class_idx],
                "confidence_percent": round(confidence, 2),
                "probabilities": {
                    "Normal": float(final_probs[0]),
                    "Alzheimer's Disease": float(final_probs[1]),
                    "Depression": float(final_probs[2])
                },
                "component_predictions": {
                    "cnn_probs": [float(p) for p in p_cnn],
                    "svm_probs": [float(p) for p in p_svm],
                    "knn_probs": [float(p) for p in p_knn]
                }
            })

        return batch_results

    def predict(self, X_3d: np.ndarray) -> np.ndarray:
        results = self.predict_detailed(X_3d)
        return np.array([r["predicted_index"] for r in results])

    def predict_proba(self, X_3d: np.ndarray) -> np.ndarray:
        results = self.predict_detailed(X_3d)
        return np.array([
            [r["probabilities"]["Normal"], r["probabilities"]["Alzheimer's Disease"], r["probabilities"]["Depression"]]
            for r in results
        ])

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.cnn.state_dict(), os.path.join(directory, "cnn_best.pth"))
        joblib.dump(self.svm.model, os.path.join(directory, "svm_model.pkl"))
        joblib.dump(self.knn.model, os.path.join(directory, "knn_model.pkl"))
        if self.meta_learner is not None:
            joblib.dump(self.meta_learner, os.path.join(directory, "meta_learner.pkl"))
        logger.info(f"Saved Hybrid Ensemble models to {directory}")

    @classmethod
    def load(cls, directory: str, n_channels: int = 19, n_features: int = 28, device: str = "cpu"):
        cnn = EEGDeepCNN(n_channels=n_channels, n_features=n_features, n_classes=3)
        cnn_path = os.path.join(directory, "cnn_best.pth")
        if os.path.exists(cnn_path):
            cnn.load_state_dict(torch.load(cnn_path, map_location=device))
        cnn.eval()

        svm = SVMClassifier()
        svm_path = os.path.join(directory, "svm_model.pkl")
        if os.path.exists(svm_path):
            svm.model = joblib.load(svm_path)

        knn = KNNClassifier()
        knn_path = os.path.join(directory, "knn_model.pkl")
        if os.path.exists(knn_path):
            knn.model = joblib.load(knn_path)

        ensemble = cls(cnn_model=cnn, svm_model=svm, knn_model=knn, device=device)
        meta_path = os.path.join(directory, "meta_learner.pkl")
        if os.path.exists(meta_path):
            ensemble.meta_learner = joblib.load(meta_path)
            ensemble.fusion_strategy = "stacking"
        return ensemble
