import random
from collections.abc import Callable
from functools import partial
from typing import Self

import numpy as np


def linear_kernel(x_i: np.ndarray, x_j: np.ndarray) -> float:
    return float(x_i @ x_j)


def polynomial_kernel(x_i: np.ndarray, x_j: np.ndarray, degree: int, gamma: float, coef0: float) -> float:
    return float((gamma * (x_i @ x_j) + coef0) ** degree)


def gaussian_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> float:
    return float(np.exp(-gamma * np.sum((x_i - x_j) ** 2)))


def linear_kernel_matrix(X: np.ndarray, x: np.ndarray) -> np.ndarray:
    return X @ x


def polynomial_kernel_matrix(X: np.ndarray, x: np.ndarray, degree: int, gamma: float, coef0: float) -> np.ndarray:
    return (gamma * (X @ x) + coef0) ** degree


def gaussian_kernel_matrix(X: np.ndarray, x: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * np.sum((X - x) ** 2, axis=-1))


#https://awstip.com/support-vector-machines-svms-a-beginners-guide-54e5c7913fc8
#https://cs229.stanford.edu/materials/smo.pdf
class SVM:
    def __init__(self,
                c: float = 1,
                max_passes : int = 3,
                tol : float = 1e-3,
                kernel : str = "linear",
                degree : int = 3,
                gamma : str | float = 'auto',
                coef0 : float = 0,
                seed = None
                ):
        self.c = c
        self.tol = tol
        self.max_passes = max_passes
        
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

        self.seed = seed

        self.alphas: np.ndarray | None = None
        self.bias: float = 0.0
        self.X_train: np.ndarray | None = None
        self.Y_train: np.ndarray | None = None
        self._kernel_fn: Callable[[np.ndarray, np.ndarray], float] | None = None
        self._kernel_matrix_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
        self._errors_cache: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> Self:
        self.X_train = X_train
        self.Y_train = np.where(Y_train == 0, -1, 1).astype(np.float64)
        self.rng = random.Random(self.seed)
        self._setup_kernel(X_train)

        m = X_train.shape[0]
        self.alphas = np.zeros(m, dtype=np.float64)
        self.bias = 0.0
        self._errors_cache = -self.Y_train.copy()
        passes = 0

        while (passes < self.max_passes):
            num_changed_alphas = 0
            for i in range(m):
                E_i = self._errors_cache[i]
                if self._violates_kkt(self.Y_train[i], E_i, self.alphas[i]):
                    j = self._random_not_equal_to(i, m)
                    E_j = self._errors_cache[j]

                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]

                    L = self._compute_alpha_lowerbound(
                        self.Y_train[i], self.Y_train[j], alpha_i_old, alpha_j_old
                    )
                    H = self._compute_alpha_upperbound(
                        self.Y_train[i], self.Y_train[j], alpha_i_old, alpha_j_old
                    )

                    if L == H:
                        continue

                    eta = self._compute_dual_objective_second_derivative(
                        X_train[i], X_train[j]
                    )
                    if eta >= 0:
                        continue

                    self.alphas[j] = self._compute_optimal_alpha_j(
                        alpha_j_old, self.Y_train[j], E_i, E_j, eta, L, H
                    )
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alphas[i] = self._compute_first_alpha_from_second(
                        alpha_i_old, alpha_j_old, self.alphas[j], self.Y_train[i], self.Y_train[j]
                    )
                    b1 = self._compute_first_bias_candidate(
                        X_train[i], X_train[j],
                        self.Y_train[i], self.Y_train[j],
                        alpha_i_old, alpha_j_old,
                        self.alphas[i], self.alphas[j],
                        E_i,
                    )
                    b2 = self._compute_second_bias_candidate(
                        X_train[i], X_train[j],
                        self.Y_train[i], self.Y_train[j],
                        alpha_i_old, alpha_j_old,
                        self.alphas[i], self.alphas[j],
                        E_j,
                    )

                    bias_old = self.bias
                    self.bias = self._resolve_bias_threshold(
                        b1, b2, self.alphas[i], self.alphas[j]
                    )
                    self._update_error_cache(
                        X_train, i, j, alpha_i_old, alpha_j_old, bias_old
                    )
                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = np.array([self._compute_decision_value(x) for x in X])
        return np.where(scores >= 0, 1, 0)

    def _setup_kernel(self, X: np.ndarray):
        gamma_value = self._resolve_gamma(X)
        if self.kernel == "linear":
            self._kernel_fn = linear_kernel
            self._kernel_matrix_fn = linear_kernel_matrix
        elif self.kernel == "poly":
            self._kernel_fn = partial(
                polynomial_kernel,
                degree=self.degree,
                gamma=gamma_value,
                coef0=self.coef0
            )
            self._kernel_matrix_fn = partial(
                polynomial_kernel_matrix,
                degree=self.degree,
                gamma=gamma_value,
                coef0=self.coef0
            )
        elif self.kernel in ("rbf", "gaussian"):
            self._kernel_fn = partial(
                gaussian_kernel,
                gamma=gamma_value
            )
            self._kernel_matrix_fn = partial(
                gaussian_kernel_matrix,
                gamma=gamma_value
            )

    def _resolve_gamma(self, X: np.ndarray) -> float:
        n_features = X.shape[1]
        if isinstance(self.gamma, (int, float)):
            return float(self.gamma)
        if self.gamma == "auto":
            return 1.0 / n_features
        elif self.gamma == "scale":
            return 1.0 / (n_features * float(X.var()))
        raise ValueError("Gamma must be 'auto', 'scale', or a numeric float")

    def _update_error_cache(
        self, X_train: np.ndarray,
        i: int, j: int,
        alpha_i_old: float, alpha_j_old: float,
        bias_old: float) -> None:
        assert self._kernel_matrix_fn is not None
        assert self.alphas is not None
        assert self.Y_train is not None
        assert self._errors_cache is not None
        k_all_i = self._kernel_matrix_fn(X_train, X_train[i])
        k_all_j = self._kernel_matrix_fn(X_train, X_train[j])
        delta_E = (
            (self.alphas[i] - alpha_i_old) * self.Y_train[i] * k_all_i
            + (self.alphas[j] - alpha_j_old) * self.Y_train[j] * k_all_j
            + (self.bias - bias_old)
        )
        self._errors_cache += delta_E

    def _compute_decision_value(self, x: np.ndarray) -> float:
        assert self.alphas is not None
        sv = np.where(self.alphas > 0)[0]
        if len(sv) == 0:
            return self.bias
        
        assert self._kernel_matrix_fn is not None
        assert self.X_train is not None
        assert self.Y_train is not None
        kernel_values = self._kernel_matrix_fn(self.X_train[sv], x)
        return float(np.sum(self.alphas[sv] * self.Y_train[sv] * kernel_values) + self.bias)

    def _compute_alpha_lowerbound(self, y_i: float, y_j: float,
                                alpha_i: float, alpha_j: float) -> float:
        if y_i != y_j:
            return max(0.0, alpha_j - alpha_i)
        return max(0, alpha_i + alpha_j - self.c)

    def _compute_alpha_upperbound(self, y_i: float, y_j: float,
                                alpha_i: float, alpha_j: float) -> float:
        if y_i != y_j:
            return min(self.c, self.c + alpha_j - alpha_i)
        return min(self.c, alpha_i + alpha_j)

    def _violates_kkt(self, y_i : float, E_i: float, alpha_i: float) -> bool:
        return ((y_i * E_i < - self.tol and alpha_i < self.c) or (y_i * E_i > self.tol and alpha_i > 0))
                
    def _random_not_equal_to(self, i : int, upper_limit:int) -> int:
        j = self.rng.randrange(upper_limit - 1)
        if j < i:
            return j
        return j + 1
    
    def _compute_dual_objective_second_derivative(self, x_i: np.ndarray, x_j: np.ndarray) -> float:
        assert self._kernel_fn is not None
        return 2 * self._kernel_fn(x_i, x_j) - self._kernel_fn(x_i, x_i) - self._kernel_fn(x_j, x_j)

    def _compute_optimal_alpha_j(self, alpha_j_old: float, y_j: float,
                            error_i: float, error_j: float, eta: float,
                            lower_bound: float, upper_bound: float) -> float:
        unclipped_alpha_j = alpha_j_old - (y_j * (error_i - error_j)) / eta

        if unclipped_alpha_j > upper_bound:
            return upper_bound
        if unclipped_alpha_j < lower_bound:
            return lower_bound
        return unclipped_alpha_j


    def _compute_first_alpha_from_second(self,
            alpha_i_old: float,
            alpha_j_old: float, alpha_j_new: float,
            y_i: float, y_j: float,
        ) -> float:
        return alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)

    def _compute_first_bias_candidate(self,
            x_i: np.ndarray, x_j: np.ndarray,
            y_i: float, y_j: float,
            alpha_i_old: float, alpha_j_old: float,
            alpha_i_new: float, alpha_j_new: float,
            error_i: float,
        ) -> float:
        assert self._kernel_fn is not None
        return float(
            self.bias - error_i
            - y_i * (alpha_i_new - alpha_i_old) * self._kernel_fn(x_i, x_i)
            - y_j * (alpha_j_new - alpha_j_old) * self._kernel_fn(x_i, x_j)
        )

    def _compute_second_bias_candidate(self,
            x_i: np.ndarray, x_j: np.ndarray,
            y_i: float, y_j: float,
            alpha_i_old: float, alpha_j_old: float,
            alpha_i_new: float, alpha_j_new: float,
            error_j: float,
        ) -> float:
        assert self._kernel_fn is not None
        return float(
            self.bias - error_j
            - y_i * (alpha_i_new - alpha_i_old) * self._kernel_fn(x_i, x_j)
            - y_j * (alpha_j_new - alpha_j_old) * self._kernel_fn(x_j, x_j)
        )

    def _resolve_bias_threshold(self, b1: float, b2: float,
                            alpha_i_new: float, alpha_j_new: float) -> float:
        if 0.0 < alpha_i_new < self.c:
            return b1
        if 0.0 < alpha_j_new < self.c:
            return b2
        return (b1 + b2) / 2.0