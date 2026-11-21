# === Implementasi algoritma CART ===

# import
import numpy as np
from typing import Any, Self
from dataclasses import dataclass

# ===== UTILS
def is_missing(val: Any) -> bool:
    return val is None or (isinstance(val, (float, np.floating)) and np.isnan(val))

# ===== STRUCTURE
@dataclass
class _Node:
    # === fitur inti ===
    value: Any
    feature_index: int | None = None # fitur yang dibandingkan
    threshold: float | None = None # batas

    left_categories: set[Any] | None = None # tipe kategorikal
    surrogates: list[tuple[int, float | set[Any], bool]] | None = None # untuk null; kalau gagal -> majority
    majority_go_left: bool = True

    left: _Node | None = None # anak node sebelah kiri
    right: _Node | None = None # anak node sebelah kanan

    # === fitur pelengkap ===
    weight_n: float = 1.0
    class_distribution: np.ndarray | None = None
    misclassification_error: float = 0.0
    gini_impurity: float = 0.0
    n_samples: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

class Cart:
    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: dict[Any, float] | str | None = None,
        ccp_alpha: float = 0.0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

        self.classes_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.root_: _Node | None = None

    # === PUBLIC
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        ...

    # === PRIVATE
    # fungsi yang dikustomisasi
    def _compute_sample_weights(self, y_idx: np.ndarray) -> np.ndarray:
        ...

    # fungsi untuk train
    # fungsi pencarian impurity terkecil untuk numerik
    def _find_best_numeric(
            self, feature: np.ndarray, target: np.ndarray, sample_weight: np.ndarray | None = None
        ) -> tuple[float | None, float, float]:
            y = np.asarray(target)
            if len(y) == 0: return None, 0.0, 0.0

            w_raw = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
            x_raw = np.asarray(feature)

            valid = np.array([not is_missing(v) for v in x_raw], dtype=bool)
            x = x_raw[valid].astype(float)
            y_v = y[valid]
            w_v = w_raw[valid]

            if len(y_v) == 0: return None, 0.0, 0.0

            p_gini = self._gini_index1(y_v, w_v)
            sorted_vals = np.sort(np.unique(x))

            if len(sorted_vals) < 2: return None, p_gini, 0.0

            thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0

            best_thresh: float | None = None
            best_weighted_gini = p_gini
            best_gain = 0.0

            for thresh in thresholds:
                left_mask = x <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                w_gini = self._gini_index2(y_v[left_mask], y_v[right_mask],
                        w_v[left_mask], w_v[right_mask])

                gain = float(p_gini - w_gini)

                if gain > best_gain:
                    best_gain = gain
                    best_thresh = float(thresh)
                    best_weighted_gini = w_gini

            return best_thresh, best_weighted_gini, best_gain

    # fungsi pencarian impurity terkecil untuk kategorikal
    def _find_best_categorical(
            self, feature: np.ndarray, target: np.ndarray, sample_weight: np.ndarray | None = None
        ) -> tuple[set[Any] | None, float, float]:
            y = np.asarray(target)
            if len(y) == 0: return None, 0.0, 0.0

            w_raw = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
            x_raw = np.asarray(feature, dtype=object)

            valid = np.array([not is_missing(v) for v in x_raw], dtype=bool)

            x_v = x_raw[valid]
            y_v = y[valid]
            w_v = w_raw[valid]

            if len(y_v) == 0: return None, 0.0, 0.0

            p_gini = self._gini_index1(y_v, w_v)
            categories = sorted(set(x_v.tolist()), key=str)
            if len(categories) < 2: return None, p_gini, 0.0

            best_left: set[Any] | None = None
            best_weighted_gini = p_gini
            best_gain = 0.0

            for cat in categories:
                left_mask = x_v == cat
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                w_gini = self._gini_index2(y_v[left_mask], y_v[right_mask],
                    w_v[left_mask], w_v[right_mask])
                gain = float(p_gini - w_gini)

                if gain > best_gain:
                    best_gain = gain
                    best_left = {cat}
                    best_weighted_gini = w_gini

            return best_left, best_weighted_gini, best_gain

    def _best_split(
            self, X: np.ndarray, y_idx: np.ndarray, sample_weight: np.ndarray
        ) -> tuple[int | None, float | None, set[Any] | None, float]:
            ...

    def _build_tree(self, X: np.ndarray, y_idx: np.ndarray, sample_weight: np.ndarray, depth: int) -> _Node:
        ...

    # prediksi test
    def _traverse(self, row: np.ndarray, node: _Node) -> Any:
        ...

    # algoritma ccp ketika tree sudah dibangun
    def _subtree_stats(self, node: _Node, total_weight: float) -> tuple[float, int]:
        ...

    def _find_weakest_link( self, node: _Node, total_weight: float) -> tuple[float, _Node | None]:
        ...

    def _prune(self, root: _Node, total_weight: float) -> None:
        ...

    # fungsi Gini_Index(T)
    @staticmethod
    def _gini_index1(y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
        y = np.asarray(y)
        if len(y) == 0: return 0.0

        w = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        total_w = float(np.sum(w))
        if total_w <= 0.0: return 0.0

        n_classes = int(np.max(y)) + 1
        class_w = np.bincount(y, weights=w, minlength=n_classes)
        prob = class_w / total_w
        return float(1.0 - np.sum(prob ** 2))

    # fungsi Gini_Index(T,A)
    @staticmethod
    def _gini_index2(
            left_y: np.ndarray,
            right_y: np.ndarray,
            left_weight: np.ndarray | None = None,
            right_weight: np.ndarray | None = None,
        ) -> float:
        w_l = np.ones(len(left_y), dtype=float) if left_weight is None else np.asarray(left_weight, dtype=float)
        w_r = np.ones(len(right_y), dtype=float) if right_weight is None else np.asarray(right_weight, dtype=float)

        sum_l = float(np.sum(w_l))
        sum_r = float(np.sum(w_r))
        total = sum_l + sum_r
        if total <= 0.0: return 0.0

        return float((sum_l / total) * Cart._gini_index1(left_y, w_l)
            + (sum_r / total) * Cart._gini_index1(right_y, w_r))
