# === Visualisasi Decision Tree (ASCII) ===
# Panggil di sini ya bung.

# import
from typing import Any
from pathlib import Path
from src.dtl.alg import Cart, _Node

# ===== UTILS
def _render_ascii(node: _Node | None, feature_names: list[str] | None,
        top_n: int, prefix: str = "", is_left: bool | None = None) -> str:
    if node is None: return ""

    # penanda cabang
    connector = ""
    if is_left is True: connector = "├── [L: True]  "
    elif is_left is False: connector = "└── [R: False] "

    lines: list[str] = []
    gini_samples = f"gini={node.gini_impurity:.3f}, samples={node.n_samples}"

    # kondisi leaf atau batas kedalaman top_n
    if node.is_leaf or top_n <= 0:
        desc = f"predict: {node.value} ({gini_samples})"
        if not node.is_leaf and top_n <= 0:
            desc += " [subtree truncated...]"
        lines.append(f"{prefix}{connector}{desc}")
        return "\n".join(lines)

    # kondisi split
    f_name = feature_names[node.feature_index] if (feature_names and node.feature_index is not None) else f"X[{node.feature_index}]"
    if node.threshold is not None: cond = f"{f_name} <= {node.threshold:.3f}"
    elif node.left_categories is not None: cond = f"{f_name} in {sorted(node.left_categories, key=str)}"
    else: cond = f"{f_name}"

    lines.append(f"{prefix}{connector}{cond} ({gini_samples})")

    # indentasi anak
    child_prefix = prefix + ("│   " if is_left is True else ("    " if is_left is False else ""))

    # rekursi kiri dan kanan
    if node.left is not None:
        lines.append(_render_ascii(node.left, feature_names, top_n - 1, child_prefix, is_left=True))
    if node.right is not None:
        lines.append(_render_ascii(node.right, feature_names, top_n - 1, child_prefix, is_left=False))

    return "\n".join(lines)

# === PUBLIC
def print_tree(model: Cart, top_n: int = 3, feature_names: list[str] | None = None,
    save_path: str | Path | None = None) -> str:
        if model.root_ is None:
            msg = "model belum di-fit."
            print(msg)
            return msg
    
        header = "=" * 60 + f"\nCART (Top-{top_n})\n" + "=" * 60
        body = _render_ascii(model.root_, feature_names, top_n)
        full_text = f"{header}\n{body}\n{'=' * 60}\n"
    
        print(full_text)
    
        if save_path is not None:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(full_text, encoding="utf-8")
            print(f"visualisasi berhasil disimpan ke: {out}")
    
        return full_text