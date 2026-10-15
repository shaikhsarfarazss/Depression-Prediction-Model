import numpy as np
from sklearn.svm import SVC
from typing import Dict, Any

class SVMClassifier:
    """
    Support Vector Machine (SVM) Classifier with RBF kernel for optimal decision hyperplanes.
    Operates on CNN deep features or preprocessed feature vectors.
    """
    def __init__(self, C: float = 10.0, kernel: str = "rbf", gamma: str = "scale", random_state: int = 42):
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            decision_function_shape="ovr",
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
