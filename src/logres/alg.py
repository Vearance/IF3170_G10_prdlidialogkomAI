from __future__ import annotations

from typing import Any

import numpy as np


class LogisticRegression:
    """Logistic Regression with Proximal Elastic Net, Dual-RBF, and Landmark Bagging."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        penalty: str | None = None,
        reg_lambda: float = 0.0001,
        l1_ratio: float = 0.5,
        use_basis: bool = False,
        continuous_cols: int = 14,
        pca_cols: int = 12,
        n_landmarks: int = 0,
        rbf_gamma: float = 0.15,
        cont_rbf_gamma: float = 0.25,
        n_estimators: int = 1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        avg_last_k: int = 1,
        threshold: float = 0.5,
        neg_weight: float = 1.0,
        pos_weight: float = 1.0,
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
        self.use_basis = use_basis
        self.continuous_cols = continuous_cols
        self.pca_cols = pca_cols
        self.n_landmarks = n_landmarks
        self.rbf_gamma = rbf_gamma
        self.cont_rbf_gamma = cont_rbf_gamma
        self.n_estimators = n_estimators
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.avg_last_k = avg_last_k
        self.threshold = threshold
        self.neg_weight = neg_weight
        self.pos_weight = pos_weight
        self.random_state = random_state

        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.triu_all_: tuple[np.ndarray, np.ndarray] | None = None
        self.triu_cont_: tuple[np.ndarray, np.ndarray] | None = None
        self.deg3_idx_: np.ndarray | None = None
        self.knots_: np.ndarray | None = None
        self.estimators_: list[dict[str, Any]] = []
        self.b: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Vectorized logistic (sigmoid) function"""
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def _log_modulus(x: np.ndarray) -> np.ndarray:
        """Log-modulus transform sign(x) * log(1 + |x|)"""
        return np.sign(x) * np.log1p(np.abs(x))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters for compatibility with sklearn.clone."""
        del deep
        return {
            name: getattr(self, name)
            for name in (
                "learning_rate",
                "max_iter",
                "penalty",
                "reg_lambda",
                "l1_ratio",
                "use_basis",
                "continuous_cols",
                "pca_cols",
                "n_landmarks",
                "rbf_gamma",
                "cont_rbf_gamma",
                "n_estimators",
                "beta1",
                "beta2",
                "eps",
                "avg_last_k",
                "threshold",
                "neg_weight",
                "pos_weight",
                "random_state",
            )
        }

    def set_params(self, **params: Any) -> LogisticRegression:
        """Set constructor parameters for compatibility with sklearn estimators."""
        valid_params = self.get_params()
        for name, value in params.items():
            if name not in valid_params:
                raise ValueError(f"Invalid parameter {name!r} for LogisticRegression.")
            setattr(self, name, value)
        return self

    def _build_deterministic_basis(self, X: np.ndarray, *, fit: bool = False) -> np.ndarray:
        """Build polynomial, cross-category, and quantile spline basis features"""
        n, d = X.shape
        d_cont = min(self.continuous_cols, d)
        d_pca = min(self.pca_cols, d)

        if fit or self.x_mean_ is None or self.x_std_ is None:
            self.x_mean_ = np.mean(X, axis=0)
            self.x_std_ = np.std(X, axis=0)
            self.x_std_[self.x_std_ == 0.0] = 1.0

        if not self.use_basis:
            return X

        if fit or self.triu_all_ is None:
            self.triu_all_ = np.triu_indices(d)
            self.triu_cont_ = np.triu_indices(d_cont)
            triples = [
                (a, b, c)
                for a in range(d_cont)
                for b in range(a, d_cont)
                for c in range(b, d_cont)
            ]
            self.deg3_idx_ = np.asarray(triples, dtype=int)
            self.knots_ = np.quantile(X[:, :d_pca], [0.1, 0.25, 0.5, 0.75, 0.9], axis=0)

        assert self.triu_all_ is not None
        assert self.triu_cont_ is not None
        assert self.deg3_idx_ is not None
        assert self.knots_ is not None

        iu0, iu1 = self.triu_all_
        inter2 = self._log_modulus(X[:, iu0] * X[:, iu1])

        idx3 = self.deg3_idx_
        inter3 = self._log_modulus(X[:, idx3[:, 0]] * X[:, idx3[:, 1]] * X[:, idx3[:, 2]])

        ic0, ic1 = self.triu_cont_
        quad_cont = self._log_modulus(X[:, ic0] * X[:, ic1])
        if d > d_cont:
            cross_cat = (quad_cont[:, :, None] * X[:, None, d_cont:]).reshape(n, -1)
        else:
            cross_cat = np.empty((n, 0), dtype=float)

        diff_knots = X[:, None, :d_pca] - self.knots_[None, :, :]
        splines_pos = np.maximum(0.0, diff_knots).reshape(n, -1)
        splines_neg = np.maximum(0.0, -diff_knots).reshape(n, -1)

        return np.hstack([X, inter2, inter3, cross_cat, splines_pos, splines_neg])

    def _build_rbf_basis(
        self, X: np.ndarray, X_norm_cont: np.ndarray, centers_raw: np.ndarray, centers_cont: np.ndarray
    ) -> np.ndarray:
        """Build RBF landmark features in full and normalized continuous spaces."""
        sq_raw = np.maximum(
            0.0,
            np.sum(X**2, axis=1, keepdims=True)
            + np.sum(centers_raw**2, axis=1)
            - 2.0 * (X @ centers_raw.T),
        )
        sq_cont = np.maximum(
            0.0,
            np.sum(X_norm_cont**2, axis=1, keepdims=True)
            + np.sum(centers_cont**2, axis=1)
            - 2.0 * (X_norm_cont @ centers_cont.T),
        )
        raw_features = np.exp(-self.rbf_gamma * sq_raw)
        cont_features = np.exp(-self.cont_rbf_gamma * sq_cont)
        return np.hstack([raw_features, cont_features])

    def _fit_single_estimator(
        self, Z: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Fit a single logistic regression weight vector using Proximal Adam + Averaging."""
        n_samples, n_features = Z.shape
        b = np.zeros(n_features, dtype=float)
        b_avg = np.zeros(n_features, dtype=float)
        m = np.zeros(n_features, dtype=float)
        v = np.zeros(n_features, dtype=float)

        k_avg = max(1, min(self.avg_last_k, self.max_iter))
        l2_coeff = 0.0
        l1_coeff = 0.0
        if self.penalty == "l2":
            l2_coeff = self.reg_lambda
        elif self.penalty == "l1":
            l1_coeff = self.reg_lambda
        elif self.penalty == "elasticnet":
            l1_coeff = self.reg_lambda * self.l1_ratio
            l2_coeff = self.reg_lambda * (1.0 - self.l1_ratio)

        sample_weights = np.where(y == 1.0, self.pos_weight, self.neg_weight)

        for t in range(1, self.max_iter + 1):
            p = self._sigmoid(Z @ b)
            grad = (Z.T @ (sample_weights * (y - p))) / n_samples

            if l2_coeff > 0.0:
                grad[1:] -= l2_coeff * b[1:]

            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad**2)
            m_hat = m / (1.0 - self.beta1**t)
            v_hat = v / (1.0 - self.beta2**t)

            eff_lr = self.learning_rate / (np.sqrt(v_hat) + self.eps)
            b += eff_lr * m_hat

            # Proximal soft-thresholding operator S_tau(b_j) for exact L1 sparsity
            if l1_coeff > 0.0:
                tau = eff_lr[1:] * l1_coeff
                b[1:] = np.sign(b[1:]) * np.maximum(0.0, np.abs(b[1:]) - tau)

            if t > self.max_iter - k_avg:
                b_avg += b / k_avg

        return b_avg

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """Fit the Multi-Seed Landmark Bagging Proximal Logistic Regression model."""
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float)
        n, d = X.shape
        d_cont = min(self.continuous_cols, d)

        base_features = self._build_deterministic_basis(X, fit=True)
        assert self.x_mean_ is not None and self.x_std_ is not None
        X_norm_cont = ((X - self.x_mean_) / self.x_std_)[:, :d_cont]

        base_seed = 42 if self.random_state is None else self.random_state
        k_centers = min(self.n_landmarks, n)
        self.estimators_ = []

        for est_idx in range(max(1, self.n_estimators)):
            if k_centers > 0:
                rng = np.random.default_rng(base_seed + est_idx * 997)
                idx_c = rng.choice(n, size=k_centers, replace=False)
                centers_raw = X[idx_c].copy()
                centers_cont = X_norm_cont[idx_c].copy()
                rbf_features = self._build_rbf_basis(X, X_norm_cont, centers_raw, centers_cont)
                phi = np.hstack([base_features, rbf_features])
            else:
                centers_raw = np.empty((0, d), dtype=float)
                centers_cont = np.empty((0, d_cont), dtype=float)
                phi = base_features

            mean = np.mean(phi, axis=0)
            std = np.std(phi, axis=0)
            std[std == 0.0] = 1.0
            Z = np.hstack([np.ones((n, 1)), (phi - mean) / std])

            b_est = self._fit_single_estimator(Z, y)
            self.estimators_.append(
                {
                    "centers_raw": centers_raw,
                    "centers_cont": centers_cont,
                    "mean": mean,
                    "std": std,
                    "b": b_est,
                }
            )

        self.b = self.estimators_[0]["b"]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate probability p = P(y=1|x) averaged across landmark-bagged estimators."""
        if not self.estimators_ or self.x_mean_ is None or self.x_std_ is None:
            raise ValueError(
                "This LogisticRegression instance is not fitted yet. "
                "Call fit() before predict()."
            )

        X_arr = np.asarray(X, dtype=float)
        n, d = X_arr.shape
        d_cont = min(self.continuous_cols, d)

        base_features = self._build_deterministic_basis(X_arr, fit=False)
        X_norm_cont = ((X_arr - self.x_mean_) / self.x_std_)[:, :d_cont]

        proba_sum = np.zeros(n, dtype=float)
        for est in self.estimators_:
            if est["centers_raw"].shape[0] > 0:
                rbf_features = self._build_rbf_basis(
                    X_arr, X_norm_cont, est["centers_raw"], est["centers_cont"]
                )
                phi = np.hstack([base_features, rbf_features])
            else:
                phi = base_features
            Z = np.hstack([np.ones((n, 1)), (phi - est["mean"]) / est["std"]])
            proba_sum += self._sigmoid(Z @ est["b"])

        return proba_sum / len(self.estimators_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class label for each row in X using threshold p >= tau."""
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)
