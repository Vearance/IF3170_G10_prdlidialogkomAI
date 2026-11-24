from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import numpy as np


# ===== UTILS
def is_missing(val: Any) -> bool:
    return val is None or (isinstance(val, (float, np.floating)) and np.isnan(val))

def valid_mask(col: np.ndarray) -> np.ndarray:
    if np.issubdtype(col.dtype, np.floating): return ~np.isnan(col)
    if np.issubdtype(col.dtype, np.integer): return np.ones(len(col), dtype=bool)
    return np.array([not is_missing(v) for v in col], dtype=bool)

def is_numeric_col(col: np.ndarray) -> bool:
    if np.issubdtype(col.dtype, np.number): return True
    non_missing = [v for v in col if not is_missing(v)]
    return len(non_missing) > 0 and all( isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
        for v in non_missing)

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

class DecisionTree:
    def __init__(
        self,
        max_depth: int | None = 15,
        min_samples_split: int = 2,
        min_samples_leaf: int = 2,
        class_weight: dict[Any, float] | str | None = {0: 1.0, 1: 1.2},
        ccp_alpha: float = 0.00010,
        max_surrogates: int = 5,
    ) -> None:
        if max_surrogates < 0:
            raise ValueError("max_surrogates harus >= 0.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_surrogates = max_surrogates

        self.classes_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.root_: _Node | None = None

    # === PUBLIC
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("model belum di-fit. panggil fit() terlebih dahulu.")

        X_arr = np.asarray(X)
        if len(X_arr) == 0:
            return np.array([])

        return np.array([self._traverse(row, self.root_) for row in X_arr])

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if len(y_arr) == 0: raise ValueError("data training tidak boleh kosong.")

        self.classes_, y_idx = np.unique(y_arr, return_inverse=True)
        self.n_features_ = X_arr.shape[1]

        sample_weight = self._compute_sample_weights(y_idx)
        total_w = float(np.sum(sample_weight))

        self.root_ = self._build_tree(X_arr, y_idx, sample_weight, depth=0)

        if self.ccp_alpha > 0.0 and self.root_ is not None:
            self._prune(self.root_, total_w)

        return self

    # === PRIVATE
    # fungsi yang dikustomisasi
    def _compute_sample_weights(self, y_idx: np.ndarray) -> np.ndarray:
        n_samples = len(y_idx)
        if n_samples == 0: return np.array([], dtype=float)

        if self.class_weight is None:
            return np.ones(n_samples, dtype=float)

        n_classes = len(self.classes_) if self.classes_ is not None else int(np.max(y_idx)) + 1

        if self.class_weight == "balanced":
            counts = np.bincount(y_idx, minlength=n_classes)
            class_w = n_samples / (n_classes * np.maximum(counts, 1))
            return class_w[y_idx]

        if isinstance(self.class_weight, dict):
            class_w = np.ones(n_classes, dtype=float)
            for idx, cls_val in enumerate(self.classes_ if self.classes_ is not None else range(n_classes)):
                class_w[idx] = float(self.class_weight.get(cls_val, 1.0))
            return class_w[y_idx]

        return np.ones(n_samples, dtype=float)

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

    # fungsi pencarian surrogate terbaik untuk numerik
    def _find_surrogate_numeric(
            self, col_valid: np.ndarray, target_left: np.ndarray, sample_weight: np.ndarray,
            tot_true: float, tot_false: float
        ) -> tuple[float | None, bool, float]:
            x_vals = col_valid.astype(float)
            order = np.argsort(x_vals, kind="quicksort")
            x_sorted = x_vals[order]
            if x_sorted[0] == x_sorted[-1]: return None, True, -1.0

            left_sorted = target_left[order]
            w_sorted = sample_weight[order]
            valid_pos = np.flatnonzero(x_sorted[:-1] != x_sorted[1:])
            if len(valid_pos) == 0: return None, True, -1.0

            # hitung kecocokan kedua arah (normal: <= ke kiri, reverse: <= ke kanan) via cumsum
            cum_left_true = np.cumsum(np.where(left_sorted, w_sorted, 0.0))[:-1]
            cum_left_false = np.cumsum(np.where(~left_sorted, w_sorted, 0.0))[:-1]
            agree_normal = (cum_left_true + (tot_false - cum_left_false))[valid_pos]
            agree_reverse = (cum_left_false + (tot_true - cum_left_true))[valid_pos]

            idx_norm = int(np.argmax(agree_normal))
            idx_rev = int(np.argmax(agree_reverse))
            if float(agree_normal[idx_norm]) >= float(agree_reverse[idx_rev]):
                best_agree = float(agree_normal[idx_norm])
                split_pos = int(valid_pos[idx_norm])
                go_left = True
            else:
                best_agree = float(agree_reverse[idx_rev])
                split_pos = int(valid_pos[idx_rev])
                go_left = False

            low = float(x_sorted[split_pos])
            high = float(x_sorted[split_pos + 1])
            thresh = low + 0.5 * (high - low)
            if not np.isfinite(thresh) or thresh >= high:
                thresh = low
            return thresh, go_left, best_agree

    # fungsi pencarian surrogate terbaik untuk kategorikal
    def _find_surrogate_categorical(
            self, col_valid: np.ndarray, target_left: np.ndarray, sample_weight: np.ndarray
        ) -> tuple[set[Any] | None, float]:
            categories = sorted(set(col_valid.tolist()), key=str)
            if len(categories) < 2: return None, -1.0

            left_cats: set[Any] = set()
            best_agree = 0.0
            for cat in categories:
                cat_mask = col_valid == cat
                w_cat_true = float(np.sum(sample_weight[cat_mask & target_left]))
                w_cat_false = float(np.sum(sample_weight[cat_mask & ~target_left]))
                if w_cat_true >= w_cat_false:
                    left_cats.add(cat)
                    best_agree += w_cat_true
                else:
                    best_agree += w_cat_false

            if len(left_cats) == 0 or len(left_cats) == len(categories):
                return None, -1.0
            return left_cats, best_agree

    # fungsi pencarian daftar surrogate terbaik (Breiman 1984)
    def _find_surrogates(
            self, X: np.ndarray, primary_feat: int, primary_valid: np.ndarray,
            primary_left: np.ndarray, sample_weight: np.ndarray
        ) -> list[tuple[int, float | set[Any], bool]]:
            if self.max_surrogates == 0: return []

            n_features = X.shape[1]
            candidates: list[tuple[float, int, float | set[Any], bool]] = []

            for f_idx in range(n_features):
                if f_idx == primary_feat: continue

                col = X[:, f_idx]
                both_valid = primary_valid & valid_mask(col)
                if not np.any(both_valid): continue

                target_left = primary_left[both_valid]
                w_valid = sample_weight[both_valid]
                tot_true = float(np.sum(w_valid[target_left]))
                tot_false = float(np.sum(w_valid[~target_left]))
                p_min = min(tot_true, tot_false)
                maj_agree = max(tot_true, tot_false)
                if p_min <= 0.0: continue

                if is_numeric_col(col):
                    split_val, go_left, best_agree = self._find_surrogate_numeric(
                        col[both_valid], target_left, w_valid, tot_true, tot_false
                    )
                else:
                    split_val, best_agree = self._find_surrogate_categorical(
                        col[both_valid], target_left, w_valid
                    )
                    go_left = True

                # hanya simpan surrogate yang lebih baik dari aturan mayoritas
                if split_val is not None and best_agree > maj_agree:
                    assoc = (best_agree - maj_agree) / p_min
                    candidates.append((assoc, f_idx, split_val, go_left))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            return [(f_idx, s_val, go_left) for _, f_idx, s_val, go_left in candidates[: self.max_surrogates]]

    def _best_split(
            self, X: np.ndarray, y_idx: np.ndarray, sample_weight: np.ndarray
        ) -> tuple[int | None, float | None, set[Any] | None, bool, float]:
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or n_samples < 2 * self.min_samples_leaf:
            return None, None, None, True, 0.0

        best_gain = 0.0
        best_feat: int | None = None
        best_thresh: float | None = None
        best_cats: set[Any] | None = None
        best_majority_left = True

        for f_idx in range(n_features):
            col = X[:, f_idx]

            # cek apakah tipe numerik atau kategorikal
            if is_numeric_col(col):
                thresh, _, gain = self._find_best_numeric(col, y_idx, sample_weight)
                if thresh is not None and gain > best_gain:
                    best_gain = gain
                    best_feat = f_idx
                    best_thresh = thresh
                    best_cats = None
                    valid = valid_mask(col)
                    left_count = np.sum(col[valid].astype(float) <= thresh)
                    best_majority_left = left_count >= (np.sum(valid) - left_count)
            else:
                cats, _, gain = self._find_best_categorical(col, y_idx, sample_weight)
                if cats is not None and gain > best_gain:
                    best_gain = gain
                    best_feat = f_idx
                    best_thresh = None
                    best_cats = cats
                    valid = valid_mask(col)
                    left_count = sum(1 for v in col[valid] if v in cats)
                    best_majority_left = left_count >= (np.sum(valid) - left_count)

        return best_feat, best_thresh, best_cats, best_majority_left, best_gain

    def _build_tree(self, X: np.ndarray, y_idx: np.ndarray, sample_weight: np.ndarray, depth: int) -> _Node:
        n_samples = len(y_idx)
        n_classes = len(self.classes_) if self.classes_ is not None else int(np.max(y_idx)) + 1

        class_w = np.bincount(y_idx, weights=sample_weight, minlength=n_classes)
        w_node = float(np.sum(class_w))
        prob = class_w / w_node if w_node > 0.0 else np.zeros(n_classes)

        # nilai mayoritas untuk prediksi leaf
        majority_idx = int(np.argmax(class_w))
        majority_val = self.classes_[majority_idx] if self.classes_ is not None else majority_idx

        gini = self._gini_index1(y_idx, sample_weight)
        misc_error = float(w_node - class_w[majority_idx])

        node = _Node(value=majority_val,weight_n=w_node,class_distribution=prob,
            misclassification_error=misc_error,gini_impurity=gini,n_samples=n_samples)

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or n_samples < 2 * self.min_samples_leaf
            or gini == 0.0
        ):
            return node

        best_feat, best_thresh, best_cats, best_maj_left, best_gain = self._best_split(X, y_idx, sample_weight)
        if best_feat is None or best_gain <= 0.0: return node

        col = X[:, best_feat]
        known_mask = valid_mask(col)
        missing_mask = ~known_mask
        left_mask = np.zeros(n_samples, dtype=bool)
        if np.any(known_mask):
            if best_thresh is not None: # fitur numerik
                left_mask[known_mask] = col[known_mask].astype(float) <= best_thresh
            else: # fitur kategorikal
                assert best_cats is not None
                left_mask[known_mask] = np.array([v in best_cats for v in col[known_mask]], dtype=bool)

        surrogates = self._find_surrogates(X, best_feat, known_mask, left_mask, sample_weight)
        if np.any(missing_mask):
            for idx in np.flatnonzero(missing_mask):
                left_mask[idx] = self._route_missing(X[idx], surrogates, best_maj_left)

        right_mask = ~left_mask

        # jika salah satu cabang kosong, jadikan leaf
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return node

        node.feature_index = best_feat
        node.threshold = best_thresh
        node.left_categories = best_cats
        node.surrogates = surrogates if surrogates else None
        node.majority_go_left = best_maj_left

        # rekursi
        node.left = self._build_tree(X[left_mask], y_idx[left_mask], sample_weight[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y_idx[right_mask], sample_weight[right_mask], depth + 1)

        return node

    # prediksi test
    def _traverse(self, row: np.ndarray, node: _Node) -> Any:
        if node.is_leaf:
            return node.value

        val = row[node.feature_index]

        if is_missing(val):
            go_left = self._route_missing(row, node.surrogates, node.majority_go_left)
            next_node = node.left if go_left else node.right
            return self._traverse(row, next_node if next_node is not None else node)

        if node.threshold is not None:
            if float(val) <= node.threshold:
                return self._traverse(row, node.left if node.left is not None else node)
            return self._traverse(row, node.right if node.right is not None else node)

        if node.left_categories is not None:
            if val in node.left_categories:
                return self._traverse(row, node.left if node.left is not None else node)
            return self._traverse(row, node.right if node.right is not None else node)

        return node.value

    # algoritma ccp ketika tree sudah dibangun
    def _subtree_stats(self, node: _Node, total_weight: float) -> tuple[float, int]:
        if node.is_leaf:
            err = (node.misclassification_error / total_weight) if total_weight > 0.0 else 0.0
            return err, 1

        err_l, leaves_l = self._subtree_stats(node.left, total_weight) if node.left is not None else (0.0, 0)
        err_r, leaves_r = self._subtree_stats(node.right, total_weight) if node.right is not None else (0.0, 0)

        return err_l + err_r, leaves_l + leaves_r

    def _find_weakest_link(self, node: _Node, total_weight: float) -> tuple[float, _Node | None]:
        if node.is_leaf: return float("inf"), None

        # hitung error jika node ini dipangkas jadi leaf vs jika dibiarkan jadi subtree
        r_t = (node.misclassification_error / total_weight) if total_weight > 0.0 else 0.0
        r_tt, n_leaves = self._subtree_stats(node, total_weight)

        alpha = (r_t - r_tt) / (n_leaves - 1) if n_leaves > 1 else float("inf")
        weakest_node: _Node | None = node

        # cek kiri
        if node.left is not None and not node.left.is_leaf:
            alpha_l, node_l = self._find_weakest_link(node.left, total_weight)
            if alpha_l < alpha:
                alpha = alpha_l
                weakest_node = node_l

        # cek kanan
        if node.right is not None and not node.right.is_leaf:
            alpha_r, node_r = self._find_weakest_link(node.right, total_weight)
            if alpha_r < alpha:
                alpha = alpha_r
                weakest_node = node_r

        return alpha, weakest_node

    def _prune(self, root: _Node, total_weight: float) -> None:
        while not root.is_leaf:
            alpha, weakest = self._find_weakest_link(root, total_weight)
            if weakest is None or alpha > self.ccp_alpha:
                break

            # pangkas node terlemah menjadi leaf
            weakest.left = None
            weakest.right = None
            weakest.feature_index = None
            weakest.threshold = None
            weakest.left_categories = None
            weakest.surrogates = None

    # fungsi bantu arahkan missing value lewat surrogate
    @staticmethod
    def _route_missing(
            row: np.ndarray, surrogates: list[tuple[int, float | set[Any], bool]] | None,
            majority_go_left: bool
        ) -> bool:
            if surrogates:
                for s_feat, s_val, go_left_when_true in surrogates:
                    val = row[s_feat]
                    if is_missing(val): continue
                    matched = val in s_val if isinstance(s_val, set) else float(val) <= float(s_val)
                    return matched if go_left_when_true else not matched
            return majority_go_left

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
    def _gini_index2(left_y: np.ndarray, right_y: np.ndarray,
            left_weight: np.ndarray | None = None, right_weight: np.ndarray | None = None) -> float:
        w_l = np.ones(len(left_y), dtype=float) if left_weight is None else np.asarray(left_weight, dtype=float)
        w_r = np.ones(len(right_y), dtype=float) if right_weight is None else np.asarray(right_weight, dtype=float)

        sum_l = float(np.sum(w_l))
        sum_r = float(np.sum(w_r))
        total = sum_l + sum_r
        if total <= 0.0: return 0.0

        return float((sum_l / total) * DecisionTree._gini_index1(left_y, w_l)
            + (sum_r / total) * DecisionTree._gini_index1(right_y, w_r))

