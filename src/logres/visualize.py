"""Generate a video/animation of the Log-Loss contour and (theta_0, theta_1) trajectory."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.driver import TARGET, TRAIN_PATH
from src.logres.alg import LogisticRegression


def _compute_log_loss(Z: np.ndarray, y: np.ndarray, b: np.ndarray) -> float:
    """Compute binary cross-entropy (log-loss) for design matrix Z = [1, x_1]."""
    p = np.clip(LogisticRegression._sigmoid(Z @ b), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _write_mjpeg_avi(frames: list[bytes], output_path: Path, fps: int = 15) -> None:
    """Write a list of JPEG-encoded frames into an uncompressed-container MJPEG .avi video."""
    if not frames:
        raise ValueError("No frames to write to AVI.")

    with Image.open(io.BytesIO(frames[0])) as img:
        width, height = img.size

    num_frames = len(frames)
    usec_per_frame = int(1_000_000 / fps)

    movi_chunks = bytearray()
    idx_entries = bytearray()
    offset = 4  # relative to 'movi' list payload start

    for jpeg_bytes in frames:
        pad = len(jpeg_bytes) % 2
        chunk_size = len(jpeg_bytes)
        movi_chunks.extend(b"00dc")
        movi_chunks.extend(struct.pack("<I", chunk_size))
        movi_chunks.extend(jpeg_bytes)
        if pad:
            movi_chunks.extend(b"\x00")

        idx_entries.extend(b"00dc")
        idx_entries.extend(struct.pack("<III", 0x10, offset, chunk_size))
        offset += 8 + chunk_size + pad

    movi_list = b"LIST" + struct.pack("<I", 4 + len(movi_chunks)) + b"movi" + movi_chunks
    idx1_chunk = b"idx1" + struct.pack("<I", len(idx_entries)) + idx_entries

    avih = struct.pack(
        "<IIIIIIIIIIIIII",
        usec_per_frame,  # dwMicroSecPerFrame
        0,               # dwMaxBytesPerSec
        0,               # dwPaddingGranularity
        0x10,            # dwFlags (AVIF_HASINDEX)
        num_frames,      # dwTotalFrames
        0,               # dwInitialFrames
        1,               # dwStreams
        0,               # dwSuggestedBufferSize
        width,           # dwWidth
        height,          # dwHeight
        0, 0, 0, 0,      # dwReserved[4]
    )
    avih_chunk = b"avih" + struct.pack("<I", len(avih)) + avih

    strh = struct.pack(
        "<4s4sIHHIIIIIIIIhhhh",
        b"vids",         # fccType
        b"MJPG",         # fccHandler
        0,               # dwFlags
        0, 0,            # wPriority, wLanguage
        0,               # dwInitialFrames
        1,               # dwScale
        fps,             # dwRate
        0,               # dwStart
        num_frames,      # dwLength
        0,               # dwSuggestedBufferSize
        0,               # dwQuality
        0,               # dwSampleSize
        0, 0, width, height,  # rcFrame
    )
    strh_chunk = b"strh" + struct.pack("<I", len(strh)) + strh

    strf = struct.pack(
        "<IIIHHIIiiii",
        40,              # biSize
        width,           # biWidth
        height,          # biHeight
        1,               # biPlanes
        24,              # biBitCount
        int.from_bytes(b"MJPG", "little"),  # biCompression
        width * height * 3,                 # biSizeImage
        0, 0, 0, 0,
    )
    strf_chunk = b"strf" + struct.pack("<I", len(strf)) + strf

    strl_payload = b"strl" + strh_chunk + strf_chunk
    strl_list = b"LIST" + struct.pack("<I", len(strl_payload)) + strl_payload

    hdrl_payload = b"hdrl" + avih_chunk + strl_list
    hdrl_list = b"LIST" + struct.pack("<I", len(hdrl_payload)) + hdrl_payload

    riff_payload = b"AVI " + hdrl_list + movi_list + idx1_chunk
    riff = b"RIFF" + struct.pack("<I", len(riff_payload)) + riff_payload

    output_path.write_bytes(riff)


def generate_contour_video(
    feature_name: str = "pca3",
    learning_rate: float = 0.08,
    max_iter: int = 120,
    output_dir: Path = Path("output/viz"),
    fps: int = 15,
) -> Path:
    """Train 1-feature Logistic Regression and render log-loss contour + trajectory .avi video."""
    train_df = pd.read_csv(TRAIN_PATH)
    x = train_df[feature_name].to_numpy(dtype=float)
    y = train_df[TARGET].to_numpy(dtype=float)
    n_samples = len(y)

    # Design matrix Z = [1, x_1] so parameters are exactly (theta_0, theta_1)
    Z = np.column_stack([np.ones(n_samples, dtype=float), x])

    # Track parameter trajectory (theta_0, theta_1) and log-loss per iteration using Adam
    theta = np.zeros(2, dtype=float)
    m = np.zeros(2, dtype=float)
    v = np.zeros(2, dtype=float)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    theta_hist = [theta.copy()]
    loss_hist = [_compute_log_loss(Z, y, theta)]

    for t in range(1, max_iter + 1):
        p = LogisticRegression._sigmoid(Z @ theta)
        grad = (Z.T @ (y - p)) / n_samples
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        theta = theta + learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        theta_hist.append(theta.copy())
        loss_hist.append(_compute_log_loss(Z, y, theta))

    traj = np.asarray(theta_hist)
    losses = np.asarray(loss_hist)

    # Build 2D log-loss contour grid around trajectory
    t0_min, t0_max = min(traj[:, 0].min(), -0.5) - 0.4, max(traj[:, 0].max(), 0.2) + 0.4
    t1_min, t1_max = min(traj[:, 1].min(), -0.2) - 0.4, max(traj[:, 1].max(), 0.2) + 0.4
    grid_t0 = np.linspace(t0_min, t0_max, 90)
    grid_t1 = np.linspace(t1_min, t1_max, 90)
    T0, T1 = np.meshgrid(grid_t0, grid_t1)

    grid_b = np.vstack([T0.ravel(), T1.ravel()])  # (2, 8100)
    logits = Z @ grid_b                           # (N, 8100)
    probs = np.clip(LogisticRegression._sigmoid(logits), 1e-12, 1.0 - 1e-12)
    grid_loss = -np.mean(
        y[:, None] * np.log(probs) + (1.0 - y[:, None]) * np.log(1.0 - probs),
        axis=0,
    ).reshape(T0.shape)

    # Set up 2-panel figure
    fig, (ax_contour, ax_loss) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=120)
    fig.suptitle(
        f"Logistic Regression Training: Log-Loss Contour & Parameter Trajectory ({feature_name})",
        fontsize=13,
        fontweight="bold",
    )

    levels = np.linspace(grid_loss.min(), np.percentile(grid_loss, 95), 28)
    cf = ax_contour.contourf(T0, T1, grid_loss, levels=levels, cmap="viridis", alpha=0.88)
    ax_contour.contour(T0, T1, grid_loss, levels=levels, colors="white", linewidths=0.45, alpha=0.5)
    fig.colorbar(cf, ax=ax_contour, label="Log-Loss")

    ax_contour.scatter(
        [traj[-1, 0]],
        [traj[-1, 1]],
        color="gold",
        edgecolor="black",
        marker="*",
        s=180,
        zorder=5,
        label=f"Converged ({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})",
    )
    (path_line,) = ax_contour.plot([], [], color="#ff4d4d", linewidth=2.2, label="Trajectory")
    (curr_dot,) = ax_contour.plot([], [], "o", color="white", markeredgecolor="red", markersize=8, zorder=6)
    info_text = ax_contour.text(
        0.03,
        0.95,
        "",
        transform=ax_contour.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax_contour.set_xlabel(r"$\theta_0$ (Intercept / Bias)")
    ax_contour.set_ylabel(rf"$\theta_1$ (Weight for {feature_name})")
    ax_contour.set_title("Log-Loss Contour Surface")
    ax_contour.legend(loc="lower right", fontsize=8)

    ax_loss.set_xlim(0, max_iter)
    ax_loss.set_ylim(losses.min() * 0.98, losses.max() * 1.02)
    ax_loss.set_xlabel("Iteration (t)")
    ax_loss.set_ylabel("Binary Cross-Entropy (Log-Loss)")
    ax_loss.set_title("Log-Loss Convergence")
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    (loss_line,) = ax_loss.plot([], [], color="#1f77b4", linewidth=2.2)
    (loss_dot,) = ax_loss.plot([], [], "o", color="#ff4d4d", markersize=6)

    plt.tight_layout()

    def init():
        path_line.set_data([], [])
        curr_dot.set_data([], [])
        loss_line.set_data([], [])
        loss_dot.set_data([], [])
        info_text.set_text("")
        return path_line, curr_dot, loss_line, loss_dot, info_text

    def update(frame_idx: int):
        path_line.set_data(traj[: frame_idx + 1, 0], traj[: frame_idx + 1, 1])
        curr_dot.set_data([traj[frame_idx, 0]], [traj[frame_idx, 1]])
        loss_line.set_data(np.arange(frame_idx + 1), losses[: frame_idx + 1])
        loss_dot.set_data([frame_idx], [losses[frame_idx]])
        info_text.set_text(
            f"Iter: {frame_idx}/{max_iter}\n"
            f"theta_0 = {traj[frame_idx, 0]:.4f}\n"
            f"theta_1 = {traj[frame_idx, 1]:.4f}\n"
            f"Log-Loss = {losses[frame_idx]:.5f}"
        )
        return path_line, curr_dot, loss_line, loss_dot, info_text

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "logres_contour_trajectory.avi"

    jpeg_frames: list[bytes] = []
    init()
    for idx in range(len(traj)):
        update(idx)
        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg", dpi=120)
        jpeg_frames.append(buf.getvalue())
    _write_mjpeg_avi(jpeg_frames, video_path, fps=fps)

    plt.close(fig)
    return video_path


if __name__ == "__main__":
    vid_path = generate_contour_video()
    print(f"Saved video: {vid_path}")
