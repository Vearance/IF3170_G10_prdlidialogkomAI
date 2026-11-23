from __future__ import annotations

from typing import Any, Self

import numpy as np

# https://github.com/maik-nack/SMO/blob/master/SVM.py
# https://www.researchgate.net/publication/2446719_Improvements_to_Platt's_SMO_Algorithm_for_SVM_Classifier_Design
# https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf

class SVM:
    def __init__(
        self,
        c: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: float | str = "scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_passes: int | None = 10,
        seed: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.c = float(c)
        self.kernel = kernel.lower()
        if self.kernel == "gaussian":
            self.kernel = "rbf"
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_passes = max_passes
        self.seed = seed
        self.max_samples = max_samples

        self.eps = 1e-7
        self.gamma_value: float = 1.0
        self.alphas: np.ndarray = np.empty(0, dtype=np.float64)
        self.f_cache: np.ndarray = np.empty(0, dtype=np.float64)
        self.i_low: int = 0
        self.i_up: int = 0
        self.b_low: float = 1.0
        self.b_up: float = -1.0

        self.i_0: np.ndarray = np.empty(0, dtype=bool)
        self.i_1: np.ndarray = np.empty(0, dtype=bool)
        self.i_2: np.ndarray = np.empty(0, dtype=bool)
        self.i_3: np.ndarray = np.empty(0, dtype=bool)
        self.i_4: np.ndarray = np.empty(0, dtype=bool)

        self.coef: np.ndarray = np.empty(0, dtype=np.float64)
        self.dual_coef: np.ndarray = np.empty(0, dtype=np.float64)
        self.threshold: float = 0.0
        self.support: np.ndarray = np.empty(0, dtype=np.int64)
        self.support_vectors: np.ndarray = np.empty((0, 0), dtype=np.float64)
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

        x_mat, y_vec, max_iter = self._initialize_fitting(x_mat, y_vec)
        iteration = 0
        num_changed = 0
        examine_all = True

        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(y_vec.shape[0]):
                    num_changed += self.examine_example(i, x_mat, y_vec)
            else:
                while True:
                    inner_success = self.take_step(
                        self.i_up, self.i_low, x_mat, y_vec
                    )
                    num_changed += int(inner_success)
                    if (
                        self.b_up > self.b_low - 2.0 * self.tol
                        or not inner_success
                    ):
                        break
                num_changed = 0

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            iteration += 1
            if iteration >= max_iter:
                break

        self._set_fitted_results(x_mat, y_vec)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_.size < 2:
            raise ValueError("Model is not fitted yet.")
        decisions = self.decision_function(X)
        return np.where(decisions >= 0.0, self.classes_[1], self.classes_[0])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        x_mat = np.asarray(X, dtype=np.float64)
        if self.kernel == "linear" and self.coef.size > 0:
            return x_mat @ self.coef - self.threshold

        scores = np.zeros(len(x_mat), dtype=np.float64)
        batch_size = 2000
        for i in range(0, len(x_mat), batch_size):
            end = min(i + batch_size, len(x_mat))
            k_batch = self._compute_kernel(x_mat[i:end], self.support_vectors)
            scores[i:end] = k_batch @ self.dual_coef

        return scores - self.threshold

    def examine_example(self, i2: int, x: np.ndarray, y: np.ndarray) -> int:
        if self.i_0[i2]:
            f2 = float(self.f_cache[i2])
        else:
            f2 = self._compute_f(x, y, i2)
            self.f_cache[i2] = f2
            if (self.i_1[i2] or self.i_2[i2]) and f2 < self.b_up:
                self.b_up = f2
                self.i_up = i2
            elif (self.i_3[i2] or self.i_4[i2]) and f2 > self.b_low:
                self.b_low = f2
                self.i_low = i2

        optimality = True
        i1 = 0
        if (
            self.i_0[i2] or self.i_1[i2] or self.i_2[i2]
        ) and self.b_low - f2 > 2.0 * self.tol:
            optimality = False
            i1 = self.i_low

        if (
            self.i_0[i2] or self.i_3[i2] or self.i_4[i2]
        ) and f2 - self.b_up > 2.0 * self.tol:
            optimality = False
            i1 = self.i_up

        if optimality:
            return 0

        if self.i_0[i2]:
            if self.b_low - f2 > f2 - self.b_up:
                i1 = self.i_low
            else:
                i1 = self.i_up

        return 1 if self.take_step(i1, i2, x, y) else 0

    def take_step(self, i1: int, i2: int, x: np.ndarray, y: np.ndarray) -> bool:
        if i1 == i2:
            return False

        y1 = float(y[i1])
        y2 = float(y[i2])
        alpha1 = float(self.alphas[i1])
        alpha2 = float(self.alphas[i2])
        f1 = float(self.f_cache[i1])
        f2 = float(self.f_cache[i2])
        s = y1 * y2

        low, high = compute_box_bounds(y1, y2, alpha1, alpha2, self.c)
        if abs(low - high) < self.eps:
            return False

        k11 = float(self._compute_kernel(x[i1 : i1 + 1], x[i1 : i1 + 1])[0, 0])
        k12 = float(self._compute_kernel(x[i1 : i1 + 1], x[i2 : i2 + 1])[0, 0])
        k22 = float(self._compute_kernel(x[i2 : i2 + 1], x[i2 : i2 + 1])[0, 0])
        eta = 2.0 * k12 - k11 - k22

        if eta < -self.eps:
            a2 = compute_unconstrained_alpha2(alpha2, y2, f1, f2, eta, low, high)
        else:
            a2 = evaluate_endpoint_objective(
                y1, y2, f1, f2, alpha1, alpha2, s, k11, k12, k22, low, high, self.eps
            )

        if abs(a2 - alpha2) < self.eps * (a2 + alpha2 + self.eps):
            return False

        a1 = alpha1 + s * (alpha2 - a2)

        if np.any(self.i_0):
            ki1 = self._compute_kernel(x[self.i_0], x[i1 : i1 + 1]).ravel()
            ki2 = self._compute_kernel(x[self.i_0], x[i2 : i2 + 1]).ravel()
            self.f_cache[self.i_0] += (
                y1 * (a1 - alpha1) * ki1 + y2 * (a2 - alpha2) * ki2
            )

        self.alphas[i1] = a1
        self.alphas[i2] = a2
        self._update_index_sets(i1, y1, a1)
        self._update_index_sets(i2, y2, a2)

        self.f_cache[i1] = (
            f1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12
        )
        self.f_cache[i2] = (
            f2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22
        )

        i_low_mask = self.i_0.copy()
        i_up_mask = self.i_0.copy()
        self._update_i_low_up_mask(i_low_mask, i_up_mask, i1)
        self._update_i_low_up_mask(i_low_mask, i_up_mask, i2)

        self.b_low, self.i_low = self._get_extreme_bound(i_low_mask, np.argmax)
        self.b_up, self.i_up = self._get_extreme_bound(i_up_mask, np.argmin)
        return True

    def _initialize_fitting(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        n_samples = y.shape[0]
        self.alphas = np.zeros(n_samples, dtype=np.float64)
        self.f_cache = np.zeros(n_samples, dtype=np.float64)

        y1_mask = y == 1.0
        self.i_1 = y1_mask.copy()
        self.i_up = int(y1_mask.nonzero()[0][0])
        self.b_up = -1.0

        y2_mask = y == -1.0
        self.i_4 = y2_mask.copy()
        self.i_low = int(y2_mask.nonzero()[0][0])
        self.b_low = 1.0

        self.f_cache[self.i_low] = 1.0
        self.f_cache[self.i_up] = -1.0

        self.i_0 = np.zeros(n_samples, dtype=bool)
        self.i_2 = np.zeros(n_samples, dtype=bool)
        self.i_3 = np.zeros(n_samples, dtype=bool)

        self.gamma_value = self._resolve_gamma(x)
        max_iter = (
            self.max_passes
            if (self.max_passes is not None and self.max_passes > 0)
            else 100
        )
        return x, y, max_iter

    def _set_fitted_results(self, x: np.ndarray, y: np.ndarray) -> None:
        self.support = np.where(self.alphas > self.eps)[0]
        self.support_vectors = x[self.support]
        self.dual_coef = self.alphas[self.support] * y[self.support]
        self.threshold = (self.b_low + self.b_up) / 2.0

        if self.kernel == "linear":
            self.coef = np.sum(
                self.dual_coef[:, None] * self.support_vectors, axis=0
            )

    def _compute_f(self, x: np.ndarray, y: np.ndarray, i: int) -> float:
        supp = np.where(self.alphas > self.eps)[0]
        if len(supp) == 0:
            return -float(y[i])

        k_row = self._compute_kernel(x[i : i + 1], x[supp]).ravel()
        margin = float(np.dot(k_row, self.alphas[supp] * y[supp]))
        return margin - float(y[i])

    def _update_index_sets(self, i: int, y: float, alpha: float) -> None:
        if self.i_0[i]:
            self.i_0[i] = False
        elif y == 1.0:
            if self.i_1[i]:
                self.i_1[i] = False
            else:
                self.i_3[i] = False
        else:
            if self.i_2[i]:
                self.i_2[i] = False
            else:
                self.i_4[i] = False

        if alpha <= self.eps or alpha >= self.c - self.eps:
            if y == 1.0:
                if alpha <= self.eps:
                    self.i_1[i] = True
                else:
                    self.i_3[i] = True
            else:
                if alpha <= self.eps:
                    self.i_4[i] = True
                else:
                    self.i_2[i] = True
        else:
            self.i_0[i] = True

    def _update_i_low_up_mask(
        self, i_low_mask: np.ndarray, i_up_mask: np.ndarray, i: int
    ) -> None:
        if self.i_3[i] or self.i_4[i]:
            i_low_mask[i] = True
        else:
            i_up_mask[i] = True

    def _get_extreme_bound(
        self, mask: np.ndarray, argfunc: Any
    ) -> tuple[float, int]:
        indices = np.where(mask)[0]
        f_vals = self.f_cache[indices]
        best_pos = int(argfunc(f_vals))
        best_idx = int(indices[best_pos])
        best_val = float(self.f_cache[best_idx])
        return best_val, best_idx

    def _compute_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            return linear_kernel(x1, x2)
        if self.kernel == "rbf":
            return rbf_kernel(x1, x2, self.gamma_value)
        if self.kernel == "poly":
            return polynomial_kernel(
                x1, x2, self.degree, self.coef0, self.gamma_value
            )
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


def compute_box_bounds(
    y1: float, y2: float, alpha1: float, alpha2: float, c: float
) -> tuple[float, float]:
    if y1 != y2:
        low = max(0.0, alpha2 - alpha1)
        high = min(c, c + alpha2 - alpha1)
    else:
        low = max(0.0, alpha2 + alpha1 - c)
        high = min(c, alpha2 + alpha1)
    return low, high


def compute_unconstrained_alpha2(
    alpha2: float,
    y2: float,
    f1: float,
    f2: float,
    eta: float,
    low: float,
    high: float,
) -> float:
    new_alpha2 = alpha2 - y2 * (f1 - f2) / eta
    if new_alpha2 < low:
        return low
    if new_alpha2 > high:
        return high
    return new_alpha2


def evaluate_endpoint_objective(
    y1: float,
    y2: float,
    f1: float,
    f2: float,
    alpha1: float,
    alpha2: float,
    s: float,
    k11: float,
    k12: float,
    k22: float,
    low: float,
    high: float,
    eps: float = 1e-7,
) -> float:
    f1_term = y1 * f1 - alpha1 * k11 - s * alpha2 * k12
    f2_term = y2 * f2 - s * alpha1 * k12 - alpha2 * k22
    l1 = alpha1 + s * (alpha2 - low)
    h1 = alpha1 + s * (alpha2 - high)
    psi_l = (
        l1 * f1_term
        + low * f2_term
        + 0.5 * (l1**2) * k11
        + 0.5 * (low**2) * k22
        + s * low * l1 * k12
    )
    psi_h = (
        h1 * f1_term
        + high * f2_term
        + 0.5 * (h1**2) * k11
        + 0.5 * (high**2) * k22
        + s * high * h1 * k12
    )

    if psi_l < psi_h - eps:
        return low
    if psi_l > psi_h + eps:
        return high
    return alpha2


def linear_kernel(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    x1_mat = np.atleast_2d(x1)
    x2_mat = np.atleast_2d(x2)
    return x1_mat @ x2_mat.T


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, gamma: float) -> np.ndarray:
    x1_mat = np.atleast_2d(x1)
    x2_mat = np.atleast_2d(x2)
    x1_norm = np.sum(x1_mat**2, axis=1, keepdims=True)
    x2_norm = np.sum(x2_mat**2, axis=1, keepdims=True)
    sq_dist = x1_norm + x2_norm.T - 2.0 * (x1_mat @ x2_mat.T)
    return np.exp(-gamma * np.maximum(sq_dist, 0.0))


def polynomial_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    degree: int,
    coef0: float,
    gamma: float = 1.0,
) -> np.ndarray:
    x1_mat = np.atleast_2d(x1)
    x2_mat = np.atleast_2d(x2)
    return (gamma * (x1_mat @ x2_mat.T) + coef0) ** degree