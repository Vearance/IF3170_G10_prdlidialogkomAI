from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.bayes.alg import NaiveBayes
from src.driver import TARGET, TRAIN_PATH


def _format_feature_label(name: str) -> str:
    words = name.replace("_", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > 18 and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines[:3])


def generate_bayes_structure_plot(
    *,
    alpha: float = 1.5,
    decision_threshold: float | None = 0.475,
    n_bins: int = 5,
    n_features: int | None = 15,
    learn_structure: bool = True,
    output_dir: Path = Path("output/viz"),
) -> Path:
    """Fit Bayes on training data and save a graph of class-feature relations."""
    train_df = pd.read_csv(TRAIN_PATH)
    feature_names = [name for name in train_df.columns if name not in {TARGET, "id"}]

    X = train_df[feature_names].to_numpy(dtype=float)
    y = train_df[TARGET].to_numpy()
    model = NaiveBayes(
        alpha=alpha,
        decision_threshold=decision_threshold,
        n_bins=n_bins,
        n_features=n_features,
        learn_structure=learn_structure,
    ).fit(X, y)

    if model.selected_features_ is None or model.feature_parents_ is None:
        raise RuntimeError("Model must be fitted before generating the graph.")

    selected_names = [
        feature_names[original_index] for original_index in model.selected_features_
    ]
    n_features = len(selected_names)
    n_columns = min(5, n_features)
    n_rows = int(np.ceil(n_features / n_columns))

    fig_width = max(10.0, 2.7 * n_columns)
    fig_height = max(7.0, 1.7 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    ax.set_xlim(-0.8, n_columns - 0.2)
    ax.set_ylim(-0.8, n_rows + 1.2)
    ax.axis("off")

    structure_title = "TAN / Bayesian Network" if learn_structure else "Naive Bayes"
    ax.set_title(
        f"Struktur Hubungan Fitur dan Variabel Kelas ({structure_title})",
        fontsize=14,
        fontweight="bold",
        color="#880E4F",
        pad=18,
    )

    class_node = (n_columns / 2 - 0.5, n_rows + 0.45)
    class_box = FancyBboxPatch(
        (class_node[0] - 0.58, class_node[1] - 0.28),
        1.16,
        0.56,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.8,
        edgecolor="#D81B60",
        facecolor="#F8BBD0",
        zorder=3,
    )
    ax.add_patch(class_box)
    ax.text(
        class_node[0],
        class_node[1],
        TARGET,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#880E4F",
        zorder=4,
    )

    positions: dict[int, tuple[float, float]] = {}
    for local_index, name in enumerate(selected_names):
        row = local_index // n_columns
        column = local_index % n_columns
        row_from_top = n_rows - 1 - row
        position = (column, row_from_top)
        positions[local_index] = position

        box = FancyBboxPatch(
            (position[0] - 0.46, position[1] - 0.27),
            0.92,
            0.54,
            boxstyle="round,pad=0.035,rounding_size=0.10",
            linewidth=1.4,
            edgecolor="#F06292",
            facecolor="#FCE4EC",
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(
            position[0],
            position[1],
            _format_feature_label(name),
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#6A1B4D",
            wrap=True,
            zorder=4,
        )

    for local_index, position in positions.items():
        arrow = FancyArrowPatch(
            (class_node[0], class_node[1] - 0.30),
            (position[0], position[1] + 0.30),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.9,
            color="#F48FB1",
            alpha=0.80,
            zorder=2,
        )
        ax.add_patch(arrow)

    for child_index, parent_index in model.feature_parents_.items():
        if parent_index is None:
            continue
        if child_index not in positions or parent_index not in positions:
            continue

        parent_position = positions[parent_index]
        child_position = positions[child_index]
        arrow = FancyArrowPatch(
            (parent_position[0], parent_position[1] - 0.30),
            (child_position[0], child_position[1] + 0.30),
            connectionstyle="arc3,rad=0.10",
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.4,
            color="#EC407A",
            zorder=5,
        )
        ax.add_patch(arrow)

    ax.text(
        -0.72,
        -0.55,
        "Panah terang: Kelas -> Fitur\nPanah gelap: Hubungan antarfitur",
        ha="left",
        va="center",
        fontsize=9,
        color="#880E4F",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "bayes_structure.png"
    output_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
