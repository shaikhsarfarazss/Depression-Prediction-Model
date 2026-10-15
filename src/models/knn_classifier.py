import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier:
    """
    K-Nearest Neighbor (KNN) Classifier for similarity-based pattern matching.
    Operates on CNN deep features or preprocessed feature vectors.
    """
    def __init__(self, n_neighbors: int = 5, weights: str = "distance", metric: str = "euclidean"):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            algorithm="auto"
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
