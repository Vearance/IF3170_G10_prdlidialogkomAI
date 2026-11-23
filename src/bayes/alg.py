from __future__ import annotations

import numpy as np


class NaiveBayes:
    """Bayesian Network Classifier berbasis DAG (Tree-Augmented / Naive Bayes).

    Referensi metode struktur pohon (TAN):
        Friedman, N., Geiger, D., & Goldszmidt, M. (1997). Bayesian Network
        Classifiers. Machine Learning, 29(2-3), 131-163.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        decision_threshold: float | None = 0.475,
        n_bins: int = 5,
        n_features: int | None = 15,
        learn_structure: bool = True,
    ) -> None:
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha (Laplace smoothing) must be positive.")
        if decision_threshold is not None and not 0.0 < decision_threshold < 1.0:
            raise ValueError("decision_threshold must be between 0 and 1.")
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        if n_features is not None and n_features < 1:
            raise ValueError("n_features must be positive.")

        self.alpha = float(alpha)
        self.decision_threshold = decision_threshold
        self.n_bins = int(n_bins)
        self.n_features = None if n_features is None else int(n_features)
        self.learn_structure = bool(learn_structure)

        self.classes_: np.ndarray | None = None
        self.class_prior_: np.ndarray | None = None
        self.selected_features_: np.ndarray | None = None
        self.bin_edges_: list[np.ndarray] | None = None
        self.feature_parents_: dict[int, int | None] | None = None
        self.cpts_: list[np.ndarray] | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> NaiveBayes:
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train)

        classes, class_indices = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        class_counts = np.bincount(class_indices, minlength=n_classes).astype(float)

        all_bin_edges = self._fit_bin_edges(X)
        X_binned_all = self._apply_bins(X, all_bin_edges)

        chi_scores = self._chi_square_scores(X_binned_all, class_indices, n_classes)
        feature_count = min(
            X.shape[1],
            X.shape[1] if self.n_features is None else self.n_features,
        )
        selected_features = np.argsort(chi_scores)[-feature_count:]

        bin_edges = [all_bin_edges[idx] for idx in selected_features]
        X_binned = X_binned_all[:, selected_features]

        feature_parents = self._learn_dag_structure(
            X_binned, class_indices, n_classes
        )
        cpts = self._fit_cpts(X_binned, class_indices, n_classes, feature_parents)

        self.classes_ = classes
        self.class_prior_ = class_counts / len(y)
        self.selected_features_ = selected_features
        self.bin_edges_ = bin_edges
        self.feature_parents_ = feature_parents
        self.cpts_ = cpts
        return self

    def predict_log_joint_proba(self, X: np.ndarray) -> np.ndarray:
        if (
            self.classes_ is None
            or self.class_prior_ is None
            or self.selected_features_ is None
            or self.bin_edges_ is None
            or self.feature_parents_ is None
            or self.cpts_ is None
        ):
            raise ValueError("Model must be fitted before calling predict.")

        X_arr = np.asarray(X, dtype=float)
        X_binned = self._apply_bins(
            X_arr[:, self.selected_features_],
            self.bin_edges_,
        )

        n_samples = len(X_arr)
        n_classes = len(self.classes_)
        joint = np.broadcast_to(
            np.log(self.class_prior_),
            (n_samples, n_classes),
        ).copy()

        for feat_idx, cpt in enumerate(self.cpts_):
            parent_idx = self.feature_parents_[feat_idx]
            child_bins = X_binned[:, feat_idx]

            if parent_idx is None:
                for class_idx in range(n_classes):
                    joint[:, class_idx] += np.log(cpt[class_idx, child_bins])
            else:
                parent_bins = X_binned[:, parent_idx]
                for class_idx in range(n_classes):
                    joint[:, class_idx] += np.log(
                        cpt[class_idx, parent_bins, child_bins]
                    )

        return joint

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        joint = self.predict_log_joint_proba(X)
        normalizer = np.max(joint, axis=1, keepdims=True)
        log_sum = normalizer + np.log(
            np.sum(np.exp(joint - normalizer), axis=1, keepdims=True)
        )
        return joint - log_sum

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Model must be fitted before calling predict.")

        if len(self.classes_) == 2 and self.decision_threshold is not None:
            probabilities = self.predict_proba(X)[:, 1]
            return self.classes_[
                (probabilities >= self.decision_threshold).astype(int)
            ]
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def _learn_dag_structure(
        self,
        X_binned: np.ndarray,
        class_indices: np.ndarray,
        n_classes: int,
    ) -> dict[int, int | None]:
        k = X_binned.shape[1]
        parents: dict[int, int | None] = {i: None for i in range(k)}

        if not self.learn_structure or k < 2:
            return parents

        cmi_matrix = np.zeros((k, k), dtype=float)
        n_samples = float(len(class_indices))

        for i in range(k):
            for j in range(i + 1, k):
                flat_idx = (
                    (class_indices * self.n_bins + X_binned[:, i]) * self.n_bins
                    + X_binned[:, j]
                )
                joint_counts = (
                    np.bincount(
                        flat_idx,
                        minlength=n_classes * self.n_bins * self.n_bins,
                    )
                    .reshape(n_classes, self.n_bins, self.n_bins)
                    .astype(float)
                )

                p_yxi_xj = joint_counts / n_samples
                p_y = p_yxi_xj.sum(axis=(1, 2), keepdims=True)
                p_yxi = p_yxi_xj.sum(axis=2, keepdims=True)
                p_yxj = p_yxi_xj.sum(axis=1, keepdims=True)

                mask = p_yxi_xj > 0
                denom = np.maximum(p_yxi * p_yxj, np.finfo(float).eps)
                ratio = (p_yxi_xj * p_y) / denom
                cmi = np.sum(p_yxi_xj[mask] * np.log(ratio[mask]))

                cmi_matrix[i, j] = cmi
                cmi_matrix[j, i] = cmi

        visited = {0}
        while len(visited) < k:
            best_weight = -1.0
            best_u, best_v = -1, -1

            for u in visited:
                for v in range(k):
                    if v not in visited and cmi_matrix[u, v] > best_weight:
                        best_weight = cmi_matrix[u, v]
                        best_u, best_v = u, v

            if best_v == -1:
                break

            visited.add(best_v)
            if best_weight > 1e-12:
                parents[best_v] = best_u

        return parents

    def _fit_cpts(
        self,
        X_binned: np.ndarray,
        class_indices: np.ndarray,
        n_classes: int,
        feature_parents: dict[int, int | None],
    ) -> list[np.ndarray]:
        cpts: list[np.ndarray] = []

        for feat_idx in range(X_binned.shape[1]):
            parent_idx = feature_parents[feat_idx]

            if parent_idx is None:
                counts = np.full(
                    (n_classes, self.n_bins), self.alpha, dtype=float
                )
                flat_idx = class_indices * self.n_bins + X_binned[:, feat_idx]
                counts += np.bincount(
                    flat_idx, minlength=n_classes * self.n_bins
                ).reshape(n_classes, self.n_bins)
                cpts.append(counts / counts.sum(axis=1, keepdims=True))
            else:
                counts = np.full(
                    (n_classes, self.n_bins, self.n_bins),
                    self.alpha,
                    dtype=float,
                )
                flat_idx = (
                    (class_indices * self.n_bins + X_binned[:, parent_idx])
                    * self.n_bins
                    + X_binned[:, feat_idx]
                )
                counts += np.bincount(
                    flat_idx,
                    minlength=n_classes * self.n_bins * self.n_bins,
                ).reshape(n_classes, self.n_bins, self.n_bins)
                cpts.append(counts / counts.sum(axis=2, keepdims=True))

        return cpts

    def _chi_square_scores(
        self,
        X_binned: np.ndarray,
        class_indices: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        scores = np.zeros(X_binned.shape[1], dtype=float)
        n_samples = float(len(class_indices))

        for feature_index in range(X_binned.shape[1]):
            flat_idx = class_indices * self.n_bins + X_binned[:, feature_index]
            contingency = (
                np.bincount(flat_idx, minlength=n_classes * self.n_bins)
                .reshape(n_classes, self.n_bins)
                .astype(float)
            )
            row_totals = contingency.sum(axis=1, keepdims=True)
            column_totals = contingency.sum(axis=0, keepdims=True)
            expected = (row_totals @ column_totals) / n_samples
            scores[feature_index] = np.sum(
                (contingency - expected) ** 2
                / np.maximum(expected, np.finfo(float).eps)
            )
        return scores

    def _fit_bin_edges(self, X: np.ndarray) -> list[np.ndarray]:
        edges: list[np.ndarray] = []
        for feature_index in range(X.shape[1]):
            values = X[:, feature_index]
            unique_values = np.unique(values)
            if len(unique_values) <= self.n_bins:
                edges.append(unique_values)
            else:
                edges.append(
                    np.quantile(
                        values,
                        np.linspace(0, 1, self.n_bins + 1)[1:-1],
                    )
                )
        return edges

    def _apply_bins(
        self,
        X: np.ndarray,
        bin_edges: list[np.ndarray],
    ) -> np.ndarray:
        X_binned: np.ndarray = np.empty(X.shape, dtype=np.int16)
        for feature_index, edges in enumerate(bin_edges):
            X_binned[:, feature_index] = np.searchsorted(
                edges,
                X[:, feature_index],
                side="right",
            )
        return np.clip(X_binned, 0, self.n_bins - 1)