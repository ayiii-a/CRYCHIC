"""Low-level deterministic slice rendering, shared by overlay.py and the web UI.

Pure matplotlib/numpy: a preprocessed intensity volume + region masks on the same
grid go in, an annotated PNG (bytes) comes out. No model, no LLM, no knowledge of
checks or thresholds — those live on the FindingCard. Kept separate from
``overlay.py`` so the FastAPI viewer and the ``render_overlay`` MCP tool draw
identical slices from one code path.
"""

from __future__ import annotations

import io
import textwrap

import matplotlib

matplotlib.use("Agg")  # headless: render to PNG bytes, never a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .geometry import PLANE_AXIS  # noqa: E402

PLANES = ("axial", "coronal", "sagittal")
# Region overlay colors (RGB 0–1).
HIPPO_RGB = (1.00, 0.36, 0.26)
VENT_RGB = (0.30, 0.80, 1.00)


def normalize(sl: np.ndarray) -> np.ndarray:
    """Percentile-clip a 2D slice to [0,1] for display."""
    fg = sl[sl > 0]
    lo, hi = (np.percentile(fg, [1, 99]) if fg.size else (0.0, 1.0))
    return np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1)


def take(vol: np.ndarray, plane: str, idx: int) -> np.ndarray:
    """One 2D slice for ``plane`` at ``idx``, oriented superior/anterior up."""
    sl = np.take(vol, idx, axis=PLANE_AXIS[plane])
    return np.rot90(sl)


def _rgba(mask2d: np.ndarray, rgb: tuple[float, float, float], alpha: float) -> np.ndarray:
    h, w = mask2d.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    out[..., 0], out[..., 1], out[..., 2] = rgb
    out[..., 3] = mask2d.astype(np.float32) * alpha
    return out


def _figure(base2d: np.ndarray, title: str, plane: str, caption: str):
    fig = plt.figure(figsize=(4.0, 4.7), dpi=120)
    fig.patch.set_facecolor("#0b1220")
    ax = fig.add_axes([0.0, 0.17, 1.0, 0.81])
    ax.imshow(base2d, cmap="gray", aspect="equal")
    ax.set_axis_off()
    ax.text(0.03, 0.97, title, color="#7dd3fc", fontsize=12, fontweight="bold",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(0.97, 0.04, plane, color="#94a3b8", fontsize=9,
            ha="right", va="bottom", transform=ax.transAxes)
    fig.text(0.03, 0.015, "\n".join(textwrap.wrap(caption, 58)) or caption,
             color="#e5e7eb", fontsize=8.0, ha="left", va="bottom")
    return fig, ax


def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def compose_region_png(
    image: np.ndarray, plane: str, idx: int,
    regions: list[tuple[np.ndarray, tuple[float, float, float], float, str]],
    title: str, caption: str,
) -> bytes:
    """Render one annotated slice: grayscale T1 + colored region overlays + legend.

    ``regions`` is a list of ``(mask3d, rgb, alpha, legend_label)``; each mask is
    sliced on the same plane/index as the image so overlays align voxel-for-voxel.
    """
    base = normalize(take(image, plane, idx))
    fig, ax = _figure(base, title, plane, caption)
    y = 0.91
    for mask3d, rgb, alpha, label in regions:
        m2d = take(mask3d, plane, idx)
        if m2d.any():
            ax.imshow(_rgba(m2d, rgb, alpha), aspect="equal")
        ax.text(0.03, y, f"■ {label}", color=tuple(rgb), fontsize=8,
                ha="left", va="top", transform=ax.transAxes)
        y -= 0.04
    return _to_png(fig)


def render_raw_png(t1_path: str, plane: str, frac: float, title: str, caption: str) -> bytes:
    """Fallback when no segmentation is available: a plain mid-ish slice, no overlay."""
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(t1_path))
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    idx = int(round(vol.shape[PLANE_AXIS[plane]] * frac))
    fig, _ = _figure(normalize(take(vol, plane, idx)), title, plane, caption)
    return _to_png(fig)
