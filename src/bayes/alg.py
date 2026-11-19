import numpy as np


class NaiveBayes:
    """Gaussian Naive Bayes classifier.
    For each class c and feature j, the model estimates
    P(xj | c) with a univariate Gaussian distribution and combines those
    likelihoods with the class prior using Bayes rule.
    """

    def __init__(
        self,
        var_smoothing: float = 0.010067,
        decision_threshold: float | None = 0.850013524074,
    ):
        if not np.isfinite(var_smoothing) or var_smoothing < 0.0:
            raise ValueError("var_smoothing must be a finite non-negative number.")
        if decision_threshold is not None and not 0.0 < decision_threshold < 1.0:
            raise ValueError("decision_threshold must be between 0 and 1.")
        self.var_smoothing = float(var_smoothing)
        self.decision_threshold = decision_threshold
        self.classes_: np.ndarray | None = None
        self.class_prior_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.var_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> NaiveBayes:
        X = self._validate_features(X_train, "X_train")
        y = np.asarray(y_train)
        if y.ndim != 1:
            raise ValueError("y_train must be a one-dimensional array.")
        if len(X) != len(y):
            raise ValueError("X_train and y_train must contain the same number of rows.")
        if len(y) == 0:
            raise ValueError("Cannot fit NaiveBayes with an empty training set.")

        classes, class_indices = np.unique(y, return_inverse=True)
        counts = np.bincount(class_indices, minlength=len(classes)).astype(float)
        global_variance = float(np.var(X, axis=0).max(initial=0.0))
        epsilon = self.var_smoothing * global_variance

        means = np.empty((len(classes), X.shape[1]), dtype=float)
        variances = np.empty_like(means)
        for class_index in range(len(classes)):
            class_rows = X[class_indices == class_index]
            means[class_index] = np.mean(class_rows, axis=0)
            variances[class_index] = np.var(class_rows, axis=0) + epsilon

        self.classes_ = classes
        self.class_prior_ = counts / len(y)
        self.theta_ = means
        self.var_ = np.maximum(variances, np.finfo(float).eps)
        return self

    def predict_log_joint_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_arr = self._validate_features(X, "X")
        assert self.class_prior_ is not None
        assert self.theta_ is not None
        assert self.var_ is not None

        log_conditional = -0.5 * (
            np.log(2.0 * np.pi * self.var_)[None, :, :]
            + ((X_arr[:, None, :] - self.theta_[None, :, :]) ** 2)
            / self.var_[None, :, :]
        )
        return np.log(self.class_prior_)[None, :] + np.sum(log_conditional, axis=2)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        joint_log_probability = self.predict_log_joint_proba(X)
        normalizer = np.max(joint_log_probability, axis=1, keepdims=True)
        log_sum = normalizer + np.log(np.sum(np.exp(joint_log_probability - normalizer), axis=1, keepdims=True))
        return joint_log_probability - log_sum

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.classes_ is not None
        if len(self.classes_) == 2 and self.decision_threshold is not None:
            probabilities = self.predict_proba(X)[:, 1]
            return self.classes_[(probabilities >= self.decision_threshold).astype(int)]
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    @staticmethod
    def _validate_features(X: np.ndarray, name: str) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional numeric array.")
        if features.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one feature.")
        if not np.isfinite(features).all():
            raise ValueError(f"{name} must contain only finite values.")
        return features

    def _check_fitted(self) -> None:
        if any(value is None for value in (self.classes_, self.class_prior_, self.theta_, self.var_)):
            raise ValueError("This NaiveBayes instance is not fitted yet. Call fit() before prediction.")
