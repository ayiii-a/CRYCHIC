"""Stage 1 — Tier-1 clinical screening (pure Python, no LLM).

Thin, warm-singleton wrapper around the vendored Xue 2024 (Nature Medicine)
ADRD model that lives in ``adrd_nmed2024/``. It turns raw UDS clinical
variables into a typed :class:`~crychic.schemas.Tier1Result` that the router
(Stage 2) consumes.

Design notes
------------
* The underlying ``adrd_tool`` is imported **lazily** and the model is loaded
  **once** (``ADRDTool`` keeps it warm). The first ``screen()`` pays the
  checkpoint-load cost; subsequent calls reuse the live model — important for
  the demo's latency budget.
* Inference is blocking CPU/GPU work, so the pipeline must use
  :func:`screen_async`, which offloads to a worker thread and keeps the asyncio
  event loop free.
* Importing ``adrd_tool`` self-configures ``sys.path`` (it adds the bundled
  ``assets/`` dir so ``import adrd`` resolves as a package). We never put
  ``assets/adrd/`` itself on the path — it ships a ``typing.py`` that would
  shadow the stdlib.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

from .schemas import AttributionHeatmap, Tier1Result

# adrd_nmed2024/ sits next to this package at the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ADRD_DIR = _REPO_ROOT / "adrd_nmed2024"

_adrd_tool: ModuleType | None = None
_tool: Any = None  # warm adrd_tool.ADRDTool instance
_lock = threading.Lock()  # guards lazy module import + model load


def _adrd_dir() -> Path:
    """Directory containing ``adrd_tool.py`` (override with ``ADRD_TOOL_DIR``)."""
    return Path(os.environ.get("ADRD_TOOL_DIR", _DEFAULT_ADRD_DIR)).expanduser()


def _patch_typing_compat() -> None:
    """Back-fill newer ``typing`` names on older Pythons (e.g. 3.10).

    The vendored Xue 2024 tree imports symbols such as ``Self`` (3.11+) straight
    from ``typing``. Copying them from ``typing_extensions`` onto the live
    ``typing`` module lets those ``from typing import ...`` statements resolve
    without editing vendored sources. No-op when the running ``typing`` already
    has them, or when ``typing_extensions`` is unavailable.
    """
    import typing

    try:
        import typing_extensions as te
    except Exception:
        return
    for name in ("Self", "Never", "LiteralString", "Unpack", "TypeVarTuple",
                 "assert_never", "assert_type", "dataclass_transform",
                 "Required", "NotRequired", "reveal_type"):
        if not hasattr(typing, name) and hasattr(te, name):
            setattr(typing, name, getattr(te, name))


def _load_adrd_tool() -> ModuleType:
    """Import the vendored ``adrd_tool`` module (idempotent)."""
    global _adrd_tool
    if _adrd_tool is None:
        d = _adrd_dir()
        if not (d / "adrd_tool.py").exists():
            raise FileNotFoundError(
                f"adrd_tool.py not found under {d}. Set ADRD_TOOL_DIR to the "
                "directory holding the vendored Xue 2024 tool."
            )
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
        _patch_typing_compat()  # vendored tree uses 3.11+ typing names
        _adrd_tool = importlib.import_module("adrd_tool")
    return _adrd_tool


@contextlib.contextmanager
def _torch_load_on(device: str):
    """Force ``torch.load`` onto an existing device for the duration.

    The bundled Tier-1 checkpoint was saved on a multi-GPU host with storages
    tagged ``cuda:1``; on a single-GPU box ``torch.load`` then raises
    "Attempting to deserialize object on CUDA device 1". The vendored
    ``from_ckpt`` loads without a ``map_location``, so we inject a valid one
    (and remap any out-of-range cuda index) *only* when it is missing — explicit
    ``map_location`` calls (e.g. the tool's own ``map_location="cpu"``) are left
    untouched. The wrapper chains to whatever ``torch.load`` is already in place
    (the vendored ``mapping`` patch) and is restored on exit.
    """
    import torch

    orig = torch.load
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0

    def _remap(ml):
        if ml is None:
            return device
        s = str(ml)
        if s.startswith("cuda:"):
            try:
                if int(s.split(":", 1)[1]) >= n:
                    return device
            except ValueError:
                pass
        return ml

    def patched(*args, **kwargs):
        if len(args) >= 2:  # map_location passed positionally
            args = (args[0], _remap(args[1]), *args[2:])
        else:
            kwargs["map_location"] = _remap(kwargs.get("map_location"))
        return orig(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = orig


def _get_tool() -> Any:
    """Return the warm ``ADRDTool`` singleton, loading the model on first use."""
    global _tool
    if _tool is None:
        with _lock:
            if _tool is None:  # double-checked under the lock
                mod = _load_adrd_tool()
                device = os.environ.get("TIER1_DEVICE", "cpu")
                # The vendored from_ckpt maps a bare "cuda" to
                # "cuda:{cuda_devices[0]}", and cuda_devices defaults to [1]
                # (never saved in the checkpoint) — an invalid ordinal on a
                # single-GPU box. Pinning to "cuda:0" skips that substitution.
                if device == "cuda":
                    device = "cuda:0"
                with _torch_load_on(device):
                    _tool = mod.ADRDTool(device=device)
    return _tool


def warmup() -> None:
    """Pre-load the model so the first real ``screen()`` is fast.

    Call this from the demo pre-cache step (see CLAUDE.md "Pre-cache the 3 demo
    cases") to move the checkpoint-load cost out of the request path.
    """
    _get_tool()


def _to_result(raw: dict) -> Tier1Result:
    """Map the ``adrd_tool.analyze`` dict onto the typed Tier-1 contract."""
    preds = raw["predictions"]
    inp = raw.get("input") or {}

    heatmap = None
    h = raw.get("heatmap")
    if h:
        files = h.get("files") or {}
        heatmap = AttributionHeatmap(
            kind=h["kind"],
            note=h["note"],
            rows=h["rows"],
            columns=h["columns"],
            values=h["values"],
            csv_path=files.get("csv"),
            png_path=files.get("png"),
        )

    return Tier1Result(
        stage_probs=preds["stage"]["probs"],
        etiology_probs=preds["etiology"]["probs"],
        all_probs=preds["all"],
        stage_top=preds["stage"]["top"],
        etiology_top=preds["etiology"]["top"],
        n_clinical_features=int(inp.get("n_clinical_features", 0)),
        imaging_used=bool(inp.get("imaging_used", False)),
        input_id=inp.get("id"),
        caveats=list(raw.get("caveats") or []),
        heatmap=heatmap,
    )


def screen(
    clinical: dict | str | Path,
    *,
    mri: str | Path | None = None,
    explain: bool = True,
    out_dir: str | Path | None = None,
) -> Tier1Result:
    """Run Tier-1 screening on one patient (blocking).

    Parameters
    ----------
    clinical:
        Raw UDS variables as a dict (e.g. ``{"NACCAGE": 80, "NACCMMSE": 22,
        "SEX": 0, "NACCNE4S": 1}``) or a path to a ``.json``/``.csv`` holding
        one record. Keys must be raw UDS variable names; unmapped keys are
        ignored and surfaced as a caveat.
    mri:
        **Optional** T1w MRI. The Xue 2024 model is multimodal and uses imaging
        only when supplied — clinicians who upload a scan get a richer screen,
        but a scan is never required. Accepts either a precomputed
        ``(1, 768, 4, 4, 4)`` SwinUNETR embedding (``.npy``, fast — the
        recommended path) or a skull-stripped, MNI-registered ``.nii.gz``
        (runs the SwinUNETR encoder on the fly, needs the SSL weights). When
        present, ``imaging_used`` is True and the heatmap gains an
        ``"MRI (img)"`` row = P(with MRI) - P(clinical-only).
    explain:
        When True, also compute the feature x label attribution heatmap. This
        runs one extra forward pass per clinical feature (leave-one-out
        occlusion); disable it for the lowest-latency path.
    out_dir:
        If given, the heatmap is also written there as ``.csv``/``.png``.

    Notes
    -----
    This is the *clinical screen* with an optional T1 embedding. The dedicated
    imaging tools (MYGO-Centiloid, MUJICA, MONAI) are separate downstream
    models and live in ``tier2_imaging``.
    """
    tool = _get_tool()
    raw = tool.analyze(
        clinical=clinical,
        mri=str(mri) if mri is not None else None,
        explain=explain,
        out_dir=out_dir,
    )
    return _to_result(raw)


async def screen_async(
    clinical: dict | str | Path,
    *,
    mri: str | Path | None = None,
    explain: bool = True,
    out_dir: str | Path | None = None,
) -> Tier1Result:
    """Async wrapper over :func:`screen` that keeps the event loop free."""
    return await asyncio.to_thread(
        screen, clinical, mri=mri, explain=explain, out_dir=out_dir
    )
