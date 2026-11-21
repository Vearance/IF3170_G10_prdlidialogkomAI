from __future__ import annotations

from typing import Any, Self

import numpy as np
from scipy.optimize import minimize


class SVMKelas:
    def __init__(
        self,
        c: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: float | str = "scale",
        coef0: float = 0.0,
        tol: float = 1e-4,
        max_samples: int | None = 2000,
        seed: int | None = None,
        max_iter: int = 200,
        **kwargs: Any,
    ) -> None:
        self.c = float(c)
        self.kernel = kernel.lower()
        if self.kernel == "gaussian":
            self.kernel = "rbf"
        if self.kernel not in {"linear", "rbf", "poly"}:
            raise ValueError(
                f"Kernel must be 'linear', 'rbf', or 'poly', got: '{kernel}'"
            )
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_samples = max_samples
        self.seed = seed
        self.max_iter = int(max_iter)

        self.eps = 1e-5
        self.gamma_value: float = 1.0
        self.alphas: np.ndarray = np.empty(0, dtype=np.float64)
        self.b: float = 0.0
        self.coef: np.ndarray = np.empty(0, dtype=np.float64)
        self.dual_coef: np.ndarray = np.empty(0, dtype=np.float64)
        self.support: np.ndarray = np.empty(0, dtype=np.int64)
        self.support_vectors: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.support_y: np.ndarray = np.empty(0, dtype=np.float64)
        self.classes_: np.ndarray = np.empty(0)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> Self:
        classes = np.unique(Y_train)
        if len(classes) != 2:
            raise ValueError(
                f"Binary classification requires 2 classes, got: {classes}"
            )
        self.classes_ = classes

        x_mat = np.asarray(X_train, dtype=np.float64)
        y_vec = np.where(Y_train == classes[1], 1.0, -1.0).astype(np.float64)

        if self.max_samples is not None and len(x_mat) > self.max_samples:
            rng = np.random.default_rng(self.seed)
            sub_indices = rng.choice(
                len(x_mat), size=self.max_samples, replace=False
            )
            x_mat = x_mat[sub_indices]
            y_vec = y_vec[sub_indices]

        self.gamma_value = self._resolve_gamma(x_mat)
        k_mat = self._compute_kernel(x_mat, x_mat)

        self.alphas = self._solve_qp(k_mat, y_vec)
        self._extract_model_parameters(x_mat, y_vec, k_mat)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_.size < 2:
            raise ValueError("Model is not fitted yet.")
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        x_mat = np.asarray(X, dtype=np.float64)
        if self.kernel == "linear" and self.coef.size > 0:
            return x_mat @ self.coef + self.b

        if self.support_vectors.shape[0] == 0:
            return np.full(len(x_mat), self.b, dtype=np.float64)

        scores = np.zeros(len(x_mat), dtype=np.float64)
        batch_size = 2000
        for i in range(0, len(x_mat), batch_size):
            end = min(i + batch_size, len(x_mat))
            k_batch = self._compute_kernel(x_mat[i:end], self.support_vectors)
            scores[i:end] = k_batch @ self.dual_coef + self.b

        return scores

    def _solve_qp(self, k_mat: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        n_samples = len(y_vec)
        p_mat = np.outer(y_vec, y_vec) * k_mat

        def objective(alpha: np.ndarray) -> float:
            return 0.5 * float(np.dot(alpha, p_mat @ alpha)) - float(np.sum(alpha))

        def gradient(alpha: np.ndarray) -> np.ndarray:
            return p_mat @ alpha - np.ones(n_samples, dtype=np.float64)

        constraints = {
            "type": "eq",
            "fun": lambda alpha: float(np.dot(alpha, y_vec)),
            "jac": lambda alpha: y_vec,
        }
        bounds = [(0.0, self.c) for _ in range(n_samples)]
        initial_alpha = np.zeros(n_samples, dtype=np.float64)

        qp_result = minimize(
            fun=objective,
            x0=initial_alpha,
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        return np.maximum(qp_result.x, 0.0)

    def _extract_model_parameters(
        self, x_mat: np.ndarray, y_vec: np.ndarray, k_mat: np.ndarray
    ) -> None:
        sv_mask = self.alphas > self.eps
        self.support = np.where(sv_mask)[0]

        if len(self.support) == 0:
            self.support = np.array([0], dtype=np.int64)

        self.support_vectors = x_mat[self.support]
        self.support_y = y_vec[self.support]
        self.dual_coef = self.alphas[self.support] * self.support_y

        free_sv = np.where((self.alphas > self.eps) & (self.alphas < self.c - self.eps))[0]
        ref_indices = free_sv if len(free_sv) > 0 else self.support

        b_values = [
            float(y_vec[k] - np.sum(self.dual_coef * k_mat[self.support, k]))
            for k in ref_indices
        ]
        self.b = float(np.mean(b_values))

        if self.kernel == "linear":
            self.coef = np.sum(
                self.dual_coef[:, None] * self.support_vectors, axis=0
            )

    def _compute_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1_mat = np.atleast_2d(x1)
        x2_mat = np.atleast_2d(x2)

        if self.kernel == "linear":
            return x1_mat @ x2_mat.T
        if self.kernel == "rbf":
            sq_norm1 = np.sum(x1_mat**2, axis=1, keepdims=True)
            sq_norm2 = np.sum(x2_mat**2, axis=1, keepdims=True)
            dist_sq = sq_norm1 + sq_norm2.T - 2.0 * (x1_mat @ x2_mat.T)
            return np.exp(-self.gamma_value * np.maximum(dist_sq, 0.0))
        if self.kernel == "poly":
            return (self.gamma_value * (x1_mat @ x2_mat.T) + self.coef0) ** self.degree
        raise ValueError(f"Unknown kernel type: '{self.kernel}'")

    def _resolve_gamma(self, x: np.ndarray) -> float:
        if isinstance(self.gamma, (int, float)):
            return float(self.gamma)
        n_features = x.shape[1]
        if self.gamma == "auto":
            return 1.0 / n_features
        if self.gamma == "scale":
            variance = float(x.var())
            return 1.0 / (n_features * variance) if variance > self.eps else 1.0
        raise ValueError(f"Unsupported gamma specification: '{self.gamma}'")


SVM = SVMKelas
