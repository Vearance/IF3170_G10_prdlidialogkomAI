from collections import Counter

import numpy as np


class KNN:
    """K-Nearest Neighbors classifier using Euclidean distance."""

    def __init__(self, n_neighbors: int = 9):
        self.n_neighbors = n_neighbors
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None


    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> KNN:
        """Store the training features and labels."""
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict a class label for each row in X."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("This KNN instance is not fitted yet. Call fit() before predict().")

        X = np.asarray(X, dtype=float)
        return np.asarray([self._predict_one(row) for row in X])

    def _predict_one(self, x: np.ndarray):
        assert self.X_train is not None and self.y_train is not None
        squared_distances = np.sum((self.X_train - x) ** 2, axis=1)
        candidates = np.argpartition(
            squared_distances, self.n_neighbors - 1
        )[: self.n_neighbors]
        cutoff_distance = squared_distances[candidates].max()
        nearest_indices = np.flatnonzero(squared_distances <= cutoff_distance)
        nearest_indices = nearest_indices[
            np.argsort(squared_distances[nearest_indices], kind="stable")[: self.n_neighbors]
        ]
        nearest_labels = self.y_train[nearest_indices].tolist()
        return Counter(nearest_labels).most_common(1)[0][0]
