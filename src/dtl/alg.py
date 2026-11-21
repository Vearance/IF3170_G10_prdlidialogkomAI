# === Implementasi algoritma CART ===

# import
import numpy as np
from typing import Any
from dataclasses import dataclass

# ===== UTILS
# fungsi Gini_Index(T)
def gini_index1(y: np.ndarray) -> float:
    y = np.asarray(y)
    if len(y) == 0: return 0.0

    _, counts = np.unique(y, return_counts=True)
    prob = counts / len(y)
    return float(1.0 - np.sum(prob ** 2))
    
# fungsi Gini_Index(T,A)
def gini_index2(left_y: np.ndarray, right_y: np.ndarray) -> float:
    n_left, n_right = len(left_y), len(right_y)
    total = n_left + n_right
    if total == 0: return 0.0

    return float((n_left / total) * gini_index1(left_y)
        + (n_right / total) * gini_index1(right_y))
    
# fungsi pencarian impurity terkecil
def find_best(feature: np.ndarray, target: np.ndarray, min_samples_leaf: int = 1) -> tuple[float, float]:
    x = np.asarray(feature, dtype=float)
    y = np.asarray(target)

    p_gini = gini_index1(y)
    if len(y) == 0: return 0.0, 0.0

    sorted_vals = np.sort(np.unique(x))
    if len(sorted_vals) < 2: return p_gini, 0.0
    
    thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0
    
    best_weighted_gini = p_gini
    best_gain = 0.0

    for thresh in thresholds: # loop di setiap threshold (batas)
        left_y = y[x <= thresh]
        right_y = y[~(x <= thresh)]

        if len(left_y) < min_samples_leaf or len(right_y) < min_samples_leaf:
            continue

        w_gini = gini_index2(left_y, right_y)
        gain = float(p_gini - w_gini)

        if gain > best_gain:
            best_gain = gain
            best_weighted_gini = w_gini

    return best_weighted_gini, best_gain

# ===== STRUCTURE
@dataclass
class _Node:
    # === fitur inti ===
    value: Any
    feature_index: int | None = None # fitur yang dibandingkan
    threshold: float | None = None # batas

    left: _Node | None = None # anak node sebelah kiri
    right: _Node | None = None # anak node sebelah kanan

    # === fitur pelengkap ===
    weight_n: float = 1.0 
    class_distribution: np.ndarray | None = None
    gini_impurity: float = 0.0
    n_samples: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None