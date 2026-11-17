import numpy as np


class LogisticRegression:
    """Logistic Regression classifier using Stochastic Gradient Ascent."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        max_iter: int = 20,
        penalty: str | None = "l2",
        reg_lambda: float = 0.001,
        l1_ratio: float = 0.5,
        random_state: int | None = 42,
    ):
        if penalty not in {None, "none", "l1", "l2", "elasticnet"}:
            raise ValueError("penalty must be None, 'none', 'l1', 'l2', or 'elasticnet'.")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be between 0.0 and 1.0.")

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.penalty = None if penalty == "none" else penalty
        self.reg_lambda = reg_lambda
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.b: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z: float | np.ndarray) -> float | np.ndarray:
        """Logistic (sigmoid) function: f(x) = 1 / (1 + e^-x)."""
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _regularization_penalty(self, weights: np.ndarray) -> np.ndarray:
        """Compute regularization derivative term for weights b_1 ... b_d."""
        if self.penalty is None or self.reg_lambda == 0.0:
            return np.zeros_like(weights)
        if self.penalty == "l1":
            return self.reg_lambda * np.sign(weights)
        if self.penalty == "l2":
            return self.reg_lambda * weights
        if self.penalty == "elasticnet":
            l1_term = self.l1_ratio * np.sign(weights)
            l2_term = (1.0 - self.l1_ratio) * weights
            return self.reg_lambda * (l1_term + l2_term)
        return np.zeros_like(weights)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegression":
        """Fit the model using Stochastic Gradient Ascent (SGA)."""
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float)

        n_samples, n_features = X.shape
        X_bias = np.hstack([np.ones((n_samples, 1)), X])

        self.b = np.zeros(n_features + 1, dtype=float)

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.max_iter):
            indices = rng.permutation(n_samples)
            for i in indices:
                xi = X_bias[i]
                yi = y[i]

                pi = float(self._sigmoid(np.dot(self.b, xi)))

                nz = xi != 0.0
                grad = np.zeros_like(self.b)
                grad[nz] = (yi - pi) * xi[nz]

                if self.penalty is not None and self.reg_lambda != 0.0:
                    reg_grad = np.zeros_like(self.b)
                    reg_grad[1:] = self._regularization_penalty(self.b[1:])
                    grad[nz] -= reg_grad[nz]

                self.b[nz] += self.learning_rate * grad[nz]

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate the probability p = P(y=1|x, b) = 1 / (1 + e^(-b^T x))."""
        if self.b is None:
            raise ValueError(
                "This LogisticRegression instance is not fitted yet. Call fit() before predict()."
            )

        X_arr = np.asarray(X, dtype=float)
        X_bias = np.hstack([np.ones((X_arr.shape[0], 1)), X_arr])
        return np.asarray(self._sigmoid(X_bias @ self.b), dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class label for each row in X using threshold p >= 0.5."""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
