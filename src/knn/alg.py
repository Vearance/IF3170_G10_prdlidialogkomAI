from __future__ import annotations

from collections import Counter

import numpy as np


class KNN:
    """K-Nearest Neighbors classifier with configurable distance metrics."""

    def __init__(
        self,
        n_neighbors: int = 7,
        weights: str = "distance",
        metric: str = "euclidean",
    ):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'.")
        if metric not in {"euclidean", "correlation"}:
            raise ValueError("metric must be 'euclidean' or 'correlation'.")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.metric_mean_: np.ndarray | None = None
        self.metric_transform_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> KNN:
        """Fit the distance metric and store the training data."""
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X_train must be 2D and y_train must be a matching 1D array.")
        if len(y) < self.n_neighbors:
            raise ValueError("n_neighbors cannot exceed the number of training rows.")
        if X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X_train must contain finite values and at least one feature.")

        self.y_train = y
        if self.metric == "euclidean":
            self.X_train = X
            self.metric_mean_ = None
            self.metric_transform_ = None
            return self

        self.metric_mean_ = X.mean(axis=0)
        centered = X - self.metric_mean_
        self.metric_transform_ = self._correlation_transform(centered, y)
        self.X_train = centered @ self.metric_transform_
        return self

    def _correlation_transform(
        self, centered_X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Scale standardized features by their target-correlation weights."""
        feature_norms = np.sqrt(np.sum(centered_X**2, axis=0))
        safe_feature_norms = np.where(feature_norms == 0.0, 1.0, feature_norms)

        classes = np.unique(y)
        class_indicators = y[:, None] == classes[None, :]
        centered_indicators = class_indicators - class_indicators.mean(axis=0)
        class_norms = np.sqrt(np.sum(centered_indicators**2, axis=0))
        safe_class_norms = np.where(class_norms == 0.0, 1.0, class_norms)
        correlations = np.abs(
            centered_X.T @ centered_indicators
            / (safe_feature_norms[:, None] * safe_class_norms[None, :])
        )
        feature_weights = correlations.max(axis=1)
        mean_weight = feature_weights.mean()
        if mean_weight > 0.0:
            feature_weights /= mean_weight
        else:
            feature_weights = np.ones_like(feature_weights)

        feature_scales = np.sqrt(np.mean(centered_X**2, axis=0))
        feature_scales = np.where(feature_scales == 0.0, 1.0, feature_scales)
        return np.diag(np.sqrt(feature_weights) / feature_scales)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict a class label for each row in X."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("This KNN instance is not fitted yet. Call fit() before predict().")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]:
            raise ValueError("X must be a 2D array with the same feature count as training data.")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values.")

        metric_mean = getattr(self, "metric_mean_", None)
        metric_transform = getattr(self, "metric_transform_", None)
        if metric_mean is not None and metric_transform is not None:
            X = (X - metric_mean) @ metric_transform

        return np.asarray([self._predict_one(row) for row in X])

    def _predict_one(self, x: np.ndarray):
        assert self.X_train is not None
        squared_distances = np.sum((self.X_train - x) ** 2, axis=1)
        return self._predict_from_squared_distances(squared_distances)

    def _predict_from_squared_distances(self, squared_distances: np.ndarray):
        assert self.y_train is not None
        candidates = np.argpartition(
            squared_distances, self.n_neighbors - 1
        )[: self.n_neighbors]
        cutoff_distance = squared_distances[candidates].max()
        nearest_indices = np.flatnonzero(squared_distances <= cutoff_distance)
        nearest_indices = nearest_indices[
            np.argsort(squared_distances[nearest_indices], kind="stable")[: self.n_neighbors]
        ]
        nearest_labels = self.y_train[nearest_indices].tolist()

        if self.weights == "uniform":
            return Counter(nearest_labels).most_common(1)[0][0]

        distances = np.sqrt(squared_distances[nearest_indices])
        exact_match = distances == 0
        if exact_match.any():
            return Counter(np.asarray(nearest_labels)[exact_match].tolist()).most_common(1)[0][0]

        class_scores: dict[object, float] = {}
        for label, distance in zip(nearest_labels, distances, strict=True):
            class_scores[label] = class_scores.get(label, 0.0) + 1.0 / distance
        return max(class_scores, key=lambda label: class_scores[label])
