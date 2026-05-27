#!/usr/bin/env python
"""CRYCHIC web UI — a thin FastAPI front end over the live pipeline.

Same pipeline the MCP tools drive (Tier-1 on the local GPU, MONAI on the local
GPU, the four LLM steps on the remote Nemotron/Claude endpoint), exposed as a
clinical single-page app instead of MCP tools:

    GET  /                          the single-page UI (webui/index.html)
    GET  /api/cases                 list the bundled OASIS-3 demo cohort
    POST /api/run?case=OAS30209     fire the pipeline for one case -> {case_id}
    GET  /api/status/{case_id}      poll stage / completed tools / elapsed
    GET  /api/evidence/{case_id}    full structured evidence + Markdown report
    GET  /api/slices/{case_id}      per-plane captions (text explanation)
    GET  /api/slices/{case_id}/{plane}.png   the captioned T1 slice image

The pipeline runs as a background asyncio task on uvicorn's event loop and writes
to the shared in-memory CaseStore; the API handlers only read it. Blocking work
(inference, slice rendering) is offloaded to worker threads so polling stays
responsive — exactly the model the MCP server uses.

Run (on the GPU box, with the LLM endpoint configured):
    NEMOTRON_URL=https://api.anthropic.com/v1/chat/completions \
    NEMOTRON_API_KEY=sk-ant-... NEMOTRON_MODEL=claude-sonnet-4-6 \
    TIER1_DEVICE=cuda TIER2_DEVICE=cuda \
    uvicorn crychic_web:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import io
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless: render to PNG bytes, never a display
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response

from crychic.cases import load_demo_cases
from crychic.pipeline import start_pipeline
from crychic.state import STORE, CaseInputs
from crychic.tier2_imaging import get_segmentation

_REPO_ROOT = Path(__file__).resolve().parent
_WEBUI = _REPO_ROOT / "webui"
_PREPRO = _REPO_ROOT / "data" / "mri_prepro"

app = FastAPI(title="CRYCHIC", docs_url="/api/docs")

# Render a slice once per (case_id, plane); volumes don't change mid-case.
_SLICE_CACHE: dict[tuple[str, str], bytes] = {}
_PLANES = ("axial", "coronal", "sagittal")
# Fractional slice positions for the raw-volume fallback (no segmentation yet).
_PLANE_FRAC = {"axial": 0.46, "coronal": 0.42, "sagittal": 0.50}
# Plane -> array axis for a RAS-oriented volume (R-L, P-A, I-S).
_PLANE_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}
# Region overlay colors (RGB 0–1): hippocampus = warm, ventricles = cool.
_HIPPO_RGB = (1.00, 0.36, 0.26)
_VENT_RGB = (0.30, 0.80, 1.00)


# ============================================================================ #
# Demo-cohort helpers
# ============================================================================ #

def _find_t1(subject: str) -> Path | None:
    hits = sorted(_PREPRO.glob(f"{subject}_*_stripped_MNI.nii.gz"))
    return hits[0] if hits else None


def _sex_letter(clinical: dict) -> str | None:
    # NACC SEX coding: 1 = male, 2 = female. Unknown codes shown raw upstream.
    return {1: "M", 2: "F"}.get(clinical.get("SEX"))


def _case_card(c) -> dict[str, Any]:
    age = c.clinical.get("NACCAGE")
    sex = _sex_letter(c.clinical)
    demo = f"{int(age)}{sex}" if (age is not None and sex) else (
        f"{int(age)}y" if age is not None else "—")
    return {
        "id": c.id,
        "subject": c.subject,
        "demo": demo,
        "mmse": c.clinical.get("NACCMMSE"),
        "apoe4": c.clinical.get("NACCNE4S"),
        "true_stage": c.true_stage,
        "true_etiologies": c.true_etiologies,
        "n_clinical": len(c.clinical),
        "has_t1": _find_t1(c.subject) is not None,
        "has_embedding": c.has_mri,
    }


# ============================================================================ #
# Slice rendering (blocking; FastAPI runs these handlers in a worker thread)
# ============================================================================ #

def _normalize(sl: np.ndarray) -> np.ndarray:
    fg = sl[sl > 0]
    lo, hi = (np.percentile(fg, [1, 99]) if fg.size else (0.0, 1.0))
    return np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1)


def _take(vol: np.ndarray, plane: str, idx: int) -> np.ndarray:
    """One 2D slice for ``plane`` at index ``idx``, oriented superior/anterior up."""
    axis = _PLANE_AXIS[plane]
    sl = np.take(vol, idx, axis=axis)
    return np.rot90(sl)


def _best_index(mask: np.ndarray, plane: str, fallback: int) -> int:
    """Slice index along ``plane`` holding the most region voxels (so the view is
    centered on the structure being analyzed), or ``fallback`` if the mask empty."""
    axis = _PLANE_AXIS[plane]
    if mask.any():
        other = tuple(i for i in range(3) if i != axis)
        return int(mask.sum(axis=other).argmax())
    return fallback


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


def _render_overlay_png(seg, plane: str, title: str, caption: str) -> bytes:
    """T1 slice with the MONAI hippocampus + ventricle regions overlaid, centered
    on the structure the view is about."""
    labels = seg.labels
    hippo = np.isin(labels, seg.hippo_idx) if seg.hippo_idx else np.zeros(labels.shape, bool)
    vent = np.isin(labels, seg.vent_idx) if seg.vent_idx else np.zeros(labels.shape, bool)

    primary = hippo if plane in ("coronal", "sagittal") else vent
    idx = _best_index(primary if primary.any() else (hippo | vent), plane,
                      seg.image.shape[_PLANE_AXIS[plane]] // 2)

    base = _normalize(_take(seg.image, plane, idx))
    fig, ax = _figure(base, title, plane, caption)
    hp, vp = _take(hippo, plane, idx), _take(vent, plane, idx)
    if hp.any():
        ax.imshow(_rgba(hp, _HIPPO_RGB, 0.55), aspect="equal")
    if vp.any():
        ax.imshow(_rgba(vp, _VENT_RGB, 0.50), aspect="equal")
    ax.text(0.03, 0.91, "■ hippocampus", color="#ff5c42", fontsize=8,
            ha="left", va="top", transform=ax.transAxes)
    ax.text(0.03, 0.87, "■ ventricles", color="#4dccff", fontsize=8,
            ha="left", va="top", transform=ax.transAxes)
    return _to_png(fig)


def _render_raw_png(t1_path: str, plane: str, title: str, caption: str) -> bytes:
    """Fallback when no segmentation is cached: a plain mid-slice, no overlay."""
    img = nib.as_closest_canonical(nib.load(t1_path))
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    idx = int(round(vol.shape[_PLANE_AXIS[plane]] * _PLANE_FRAC[plane]))
    fig, _ = _figure(_normalize(_take(vol, plane, idx)), title, plane, caption)
    return _to_png(fig)


def _slice_caption(plane: str, anatomy) -> str:
    """Evidence-derived text explanation for each view (hedged, CDS-style)."""
    if anatomy is None:
        return {
            "axial": "Lateral ventricles — structural segmentation unavailable "
                     "for this case; assess ventricle size visually.",
            "coronal": "Medial temporal lobe — segmentation unavailable; assess "
                       "hippocampal volume visually.",
            "sagittal": "Midsagittal reference — global atrophy context.",
        }[plane]
    if plane == "coronal":
        z = anatomy.hippocampus_zscore
        ztxt = f"Z {z}" if z is not None else "Z n/a"
        tot = anatomy.hippocampus_total_mm3
        tottxt = f"{tot:.0f} mm³" if tot else "n/a"
        return (f"Hippocampus (red) — total {tottxt}, {ztxt} "
                f"(≤ -1.5 = atrophy vs internal norm). Dominant atrophy: "
                f"{anatomy.dominant_atrophy}. MONAI wholeBrainSeg.")
    if plane == "axial":
        vi = anatomy.ventricular_index
        return (f"Lateral ventricles (blue) — ventricular index "
                f"{vi if vi is not None else 'n/a'} (ventricle ÷ brain volume); "
                f"{anatomy.n_labels} structures segmented. MONAI wholeBrainSeg.")
    return ("Midsagittal reference — hippocampus (red) and ventricles (blue) in "
            "context; correlate with the per-region metrics on the other views.")


# ============================================================================ #
# Routes
# ============================================================================ #

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_WEBUI / "index.html").read_text(encoding="utf-8")


@app.get("/api/cases")
def list_cases() -> dict[str, Any]:
    return {"cases": [_case_card(c) for c in load_demo_cases()]}


@app.post("/api/run")
async def run_case(case: str) -> dict[str, Any]:
    # async so this runs on the event loop: start_pipeline -> asyncio.create_task
    # needs a running loop, which FastAPI's threadpool (sync defs) does not have.
    matches = [c for c in load_demo_cases()
               if case in c.id or case in c.subject]
    if not matches:
        raise HTTPException(404, f"no demo case matching {case!r}")
    c = matches[0]
    t1 = _find_t1(c.subject)
    state = start_pipeline(CaseInputs(
        clinical=c.clinical,
        t1_path=str(t1) if t1 else None,
        mri_embedding_path=str(c.embedding_path) if c.embedding_path else None,
    ))
    return {"case_id": state.case_id, "case": _case_card(c)}


@app.get("/api/status/{case_id}")
def status(case_id: str) -> dict[str, Any]:
    state = STORE.get(case_id)
    if state is None:
        raise HTTPException(404, "unknown case_id")
    return state.status()


@app.get("/api/evidence/{case_id}")
def evidence(case_id: str) -> JSONResponse:
    state = STORE.get(case_id)
    if state is None:
        raise HTTPException(404, "unknown case_id")
    return JSONResponse(state.evidence().model_dump(exclude_none=True))


@app.get("/api/slices/{case_id}")
def slice_meta(case_id: str) -> dict[str, Any]:
    state = STORE.get(case_id)
    if state is None:
        raise HTTPException(404, "unknown case_id")
    anatomy = state.tier2.anatomy if state.tier2 else None
    has_t1 = bool(state.inputs.t1_path)
    titles = {"axial": "Ventricular level", "coronal": "Hippocampal level",
              "sagittal": "Midline"}
    return {
        "has_t1": has_t1,
        "slices": [
            {"plane": p, "title": titles[p],
             "caption": _slice_caption(p, anatomy),
             "url": f"/api/slices/{case_id}/{p}.png" if has_t1 else None}
            for p in _PLANES
        ],
    }


@app.get("/api/slices/{case_id}/{plane}.png")
def slice_png(case_id: str, plane: str) -> Response:
    if plane not in _PLANES:
        raise HTTPException(404, "unknown plane")
    state = STORE.get(case_id)
    if state is None:
        raise HTTPException(404, "unknown case_id")
    t1 = state.inputs.t1_path
    if not t1:
        raise HTTPException(404, "no T1 volume for this case")

    key = (case_id, plane)
    if key not in _SLICE_CACHE:
        anatomy = state.tier2.anatomy if state.tier2 else None
        titles = {"axial": "Ventricular level", "coronal": "Hippocampal level",
                  "sagittal": "Midline"}
        caption = _slice_caption(plane, anatomy)
        seg = get_segmentation(t1)
        _SLICE_CACHE[key] = (
            _render_overlay_png(seg, plane, titles[plane], caption) if seg
            else _render_raw_png(t1, plane, titles[plane], caption))
    return Response(_SLICE_CACHE[key], media_type="image/png")
