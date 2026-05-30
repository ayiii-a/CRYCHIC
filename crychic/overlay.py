"""Deterministic annotation rendering — the "annotated key image" of a finding card.

For a structural check, locate the region the metric is about in the cached
segmentation, pick the slice that best shows it, highlight it on the
(grid-aligned) T1, and write a PNG. The number / threshold / reference live on the
FindingCard text (S4d); this module only produces the verifiable image. No model
inference here — it reads the segmentation :mod:`crychic.segmentation` already
cached (Inv #6) and draws via :mod:`crychic.render`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from . import checks, geometry, render, segmentation
from .schemas import ImagingCheck

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _overlay_dir(out_dir: str | None) -> Path:
    d = Path(out_dir) if out_dir else Path(
        os.environ.get("CRYCHIC_OVERLAY_DIR", _REPO_ROOT / ".crychic_overlays"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _regions_for(seg: segmentation.SegResult, spec: "checks.CheckSpec"):
    """(primary mask, list of (mask, rgb, alpha, label)) for a check's overlay.

    The highlighted labels come from the precomputed ``SegResult`` index list named
    by ``spec.region_attr`` (e.g. ``hippo_idx``), so the overlay aligns with the
    geometry that produced the number.
    """
    labels = seg.labels
    idxs = getattr(seg, spec.region_attr, None) or []
    mask = np.isin(labels, idxs) if idxs else np.zeros(labels.shape, bool)
    return mask, [(mask, spec.overlay_rgb, spec.overlay_alpha, spec.overlay_label)]


def render_overlay(t1_path: str, check: ImagingCheck, *, out_dir: str | None = None) -> dict | None:
    """Render the annotated key slice for ``check``; return {png_path, plane, index}.

    Returns ``None`` if the T1 has not been segmented (no fabricated image) or the
    check has no T1 overlay (e.g. FAZEKAS is a FLAIR finding — ``spec.plane`` None).
    """
    spec = checks.CHECKS.get(check)
    if spec is None or spec.plane is None or spec.region_attr is None:
        return None
    seg = segmentation.get_segmentation(t1_path)
    if seg is None:
        return None

    plane = spec.plane
    primary, regions = _regions_for(seg, spec)
    idx = geometry.pick_key_slice(primary, plane)
    caption = (f"{spec.overlay_title} — region highlighted for verification; the "
               "value, threshold and reference are on the finding card.")
    png = render.compose_region_png(seg.image, plane, idx, regions,
                                    spec.overlay_title, caption)

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(t1_path).name) or "case"
    path = _overlay_dir(out_dir) / f"{stem}_{check.value}_{plane}{idx}.png"
    path.write_bytes(png)
    return {"png_path": str(path), "plane": plane, "index": int(idx)}
