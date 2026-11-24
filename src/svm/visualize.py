"""Visualize SVM decision boundary, margins, and support vectors for Linear, Polynomial, and Gaussian kernels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.driver import TARGET, TRAIN_PATH
from src.svm.alg import SVM


def _sample_stratified(
    X: np.ndarray, y: np.ndarray, n_per_class: int = 120, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a balanced 2D subset so SMO converges quickly and points remain readable"""
    rng = np.random.default_rng(seed)
    idx_0 = rng.choice(np.flatnonzero(y == 0), size=n_per_class, replace=False)
    idx_1 = rng.choice(np.flatnonzero(y == 1), size=n_per_class, replace=False)
    idx = np.concatenate([idx_0, idx_1])
    rng.shuffle(idx)
    return X[idx], y[idx]


def generate_svm_kernel_plots(
    features: tuple[str, str] | None = None,
    n_per_class: int = 120,
    output_dir: Path = Path("output/viz"),
    seed: int = 42,
    kernel: str | None = None,
    c: float = 1.0,
    degree: int = 3,
    gamma: float | str = "scale",
    coef0: float = 0.0,
    max_passes: int = 10,
    tol: float = 1e-3,
) -> Path:
    """Train SVM on 2D features and save decision boundary, margin, and support vector plot(s)"""
    train_df = pd.read_csv(TRAIN_PATH)
    if features is None:
        corr_series = pd.Series(train_df.corr(numeric_only=True)[TARGET]).drop(TARGET).abs()
        top_two = [str(col) for col in corr_series.nlargest(2).index]
        resolved_features: tuple[str, str] = (top_two[0], top_two[1])
    else:
        resolved_features = features

    X_full = train_df[list(resolved_features)].to_numpy(dtype=float)
    y_full = train_df[TARGET].to_numpy(dtype=int)

    X_sub, y_sub = _sample_stratified(X_full, y_full, n_per_class=n_per_class, seed=seed)

    # Standardize for numerically stable kernels, while plotting in original feature coordinates
    x_mean = X_sub.mean(axis=0)
    x_std = X_sub.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    X_scaled = (X_sub - x_mean) / x_std

    x0_min, x0_max = np.percentile(X_sub[:, 0], [1, 99])
    x1_min, x1_max = np.percentile(X_sub[:, 1], [1, 99])
    pad0 = 0.20 * max(x0_max - x0_min, 1e-6)
    pad1 = 0.20 * max(x1_max - x1_min, 1e-6)

    xx, yy = np.meshgrid(
        np.linspace(x0_min - pad0, x0_max + pad0, 140),
        np.linspace(x1_min - pad1, x1_max + pad1, 140),
    )
    grid_raw = np.column_stack([xx.ravel(), yy.ravel()])
    grid_scaled = (grid_raw - x_mean) / x_std

    all_kernels: dict[str, tuple[str, dict[str, Any]]] = {
        "linear": (
            f"Linear Kernel (C={c})",
            {"c": c, "kernel": "linear", "max_passes": max_passes, "tol": tol, "seed": seed},
        ),
        "poly": (
            f"Polynomial Kernel (deg={degree}, gamma={gamma}, C={c})",
            {
                "c": c,
                "kernel": "poly",
                "degree": degree,
                "gamma": gamma,
                "coef0": coef0,
                "max_passes": max_passes,
                "tol": tol,
                "seed": seed,
            },
        ),
        "rbf": (
            f"Gaussian (RBF) Kernel (gamma={gamma}, C={c})",
            {"c": c, "kernel": "gaussian", "gamma": gamma, "max_passes": max_passes, "tol": tol, "seed": seed},
        ),
    }

    if kernel is None:
        kernel_configs = [all_kernels["linear"], all_kernels["poly"], all_kernels["rbf"]]
        filename = "svm_kernels_2d.png"
    else:
        norm_kernel = "rbf" if kernel.lower() in {"rbf", "gaussian"} else kernel.lower()
        if norm_kernel not in all_kernels:
            raise ValueError(f"Unsupported kernel for visualization: {kernel}")
        kernel_configs = [all_kernels[norm_kernel]]
        filename = f"svm_{norm_kernel}_2d.png"

    n_plots = len(kernel_configs)
    fig, axes_raw = plt.subplots(
        1, n_plots, figsize=(5.8 * n_plots, 5.2), dpi=140, sharex=True, sharey=True
    )
    axes = [axes_raw] if n_plots == 1 else list(axes_raw)
    fig.suptitle(
        f"SVM Decision Boundary, Margins, and Support Vectors ({resolved_features[0]} vs {resolved_features[1]})",
        fontsize=13.5,
        fontweight="bold",
    )

    for ax, (title, cfg) in zip(axes, kernel_configs, strict=True):
        model = SVM(**cfg).fit(X_scaled, y_sub)
        zz = model.decision_function(grid_scaled).reshape(xx.shape)

        sv_raw = model.support_vectors * x_std + x_mean
        n_sv = len(sv_raw)

        ax.contourf(xx, yy, zz, levels=[-1e9, 0.0, 1e9], colors=["#cfe2f3", "#f4cccc"], alpha=0.55)

        # Margins (-1, +1) and separating hyperplane (0)
        cs = ax.contour(
            xx,
            yy,
            zz,
            levels=[-1.0, 0.0, 1.0],
            colors=["#333333", "#000000", "#333333"],
            linestyles=["--", "-", "--"],
            linewidths=[1.4, 2.2, 1.4],
        )
        ax.clabel(cs, fmt={-1.0: "-1", 0.0: "0", 1.0: "+1"}, fontsize=8, inline=True)

        # Training samples by class
        ax.scatter(
            X_sub[y_sub == 0, 0],
            X_sub[y_sub == 0, 1],
            c="#1f77b4",
            s=28,
            alpha=0.8,
            label="Not Canceled (0)",
        )
        ax.scatter(
            X_sub[y_sub == 1, 0],
            X_sub[y_sub == 1, 1],
            c="#d62728",
            s=28,
            alpha=0.8,
            label="Canceled (1)",
        )

        # Highlight support vectors
        if n_sv > 0:
            ax.scatter(
                sv_raw[:, 0],
                sv_raw[:, 1],
                s=95,
                facecolors="none",
                edgecolors="black",
                linewidths=1.3,
                label=f"Support Vectors ({n_sv})",
            )

        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlabel(resolved_features[0])
        ax.set_xlim(x0_min - pad0, x0_max + pad0)
        ax.set_ylim(x1_min - pad1, x1_max + pad1)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[0].set_ylabel(resolved_features[1])
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = generate_svm_kernel_plots()
    print(f"Saved SVM visualization to: {path}")
