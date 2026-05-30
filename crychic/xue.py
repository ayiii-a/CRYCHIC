"""S2 — the dementia differential (Xue 2024 / Nature Medicine), multimodal.

Warm-singleton wrapper around the vendored Xue 2024 ADRD model in
``adrd_nmed2024/``. It turns raw UDS clinical variables (and, when available, the
precomputed SwinUNETR MRI embedding) into a typed
:class:`~crychic.schemas.Differential` that the router (S3) consumes.

**The MRI embedding is fed to the model when one is supplied** (``mri=`` →
``adrd_tool.analyze``): the differential is then multimodal and ``imaging_used`` is
set on the result. Because the model has already seen the MRI, the downstream
structural finding cards are a *consistency cross-check on imaging-informed
probabilities*, not a fully independent second opinion — the report wording in
``agent/reasoner`` reflects that. The geometry biomarkers themselves are still
computed independently of the model (their numbers come from ``geometry.py``).

Design notes
------------
* ``adrd_tool`` is imported lazily and the model is loaded once (kept warm). The
  first ``screen()`` pays the checkpoint-load cost; later calls reuse the live
  model — important for the demo's latency budget (Inv #3).
* Inference is blocking CPU/GPU work, so the pipeline uses :func:`screen_async`,
  which offloads to a worker thread and keeps the event loop free.
* Importing ``adrd_tool`` self-configures ``sys.path`` (it adds the bundled
  ``assets/`` dir so ``import adrd`` resolves). We never put ``assets/adrd/`` on
  the path — it ships a ``typing.py`` that would shadow the stdlib.
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

from .schemas import AttributionHeatmap, Differential

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ADRD_DIR = _REPO_ROOT / "adrd_nmed2024"

_adrd_tool: ModuleType | None = None
_tool: Any = None  # warm adrd_tool.ADRDTool instance
_lock = threading.Lock()


def _adrd_dir() -> Path:
    """Directory containing ``adrd_tool.py`` (override with ``ADRD_TOOL_DIR``)."""
    return Path(os.environ.get("ADRD_TOOL_DIR", _DEFAULT_ADRD_DIR)).expanduser()


def _patch_typing_compat() -> None:
    """Back-fill newer ``typing`` names on older Pythons so vendored sources import.

    No-op when the running ``typing`` already has them or ``typing_extensions`` is
    unavailable.
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
                "directory holding the vendored Xue 2024 tool.")
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
        _patch_typing_compat()
        _adrd_tool = importlib.import_module("adrd_tool")
    return _adrd_tool


@contextlib.contextmanager
def _torch_load_on(device: str):
    """Force ``torch.load`` onto an existing device for the duration.

    The bundled checkpoint was saved on a multi-GPU host with storages tagged
    ``cuda:1``; on a single-GPU box ``torch.load`` then raises. We inject a valid
    ``map_location`` (and remap out-of-range cuda indices) only when one is missing.
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
        if len(args) >= 2:
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
            if _tool is None:
                mod = _load_adrd_tool()
                device = os.environ.get("TIER1_DEVICE", "cpu")
                if device == "cuda":
                    device = "cuda:0"
                with _torch_load_on(device):
                    _tool = mod.ADRDTool(device=device)
    return _tool


def warmup() -> None:
    """Pre-load the model so the first real ``screen()`` is inference-only (Inv #3)."""
    _get_tool()


def _to_result(raw: dict) -> Differential:
    """Map the ``adrd_tool.analyze`` dict onto the typed S2 contract.

    When an MRI embedding was supplied, ``input.imaging_used`` is True and the
    attribution heatmap carries an extra "MRI (img)" row = P(with MRI) −
    P(clinical-only); that row is passed through unchanged.
    """
    preds = raw["predictions"]
    inp = raw.get("input") or {}

    heatmap = None
    h = raw.get("heatmap")
    if h:
        files = h.get("files") or {}
        heatmap = AttributionHeatmap(
            kind=h["kind"], note=h["note"], rows=h["rows"], columns=h["columns"],
            values=h["values"], csv_path=files.get("csv"), png_path=files.get("png"),
        )

    return Differential(
        stage_probs=preds["stage"]["probs"],
        etiology_probs=preds["etiology"]["probs"],
        all_probs=preds["all"],
        stage_top=preds["stage"]["top"],
        etiology_top=preds["etiology"]["top"],
        imaging_used=bool(inp.get("imaging_used")),
        n_clinical_features=int(inp.get("n_clinical_features", 0)),
        input_id=inp.get("id"),
        caveats=list(raw.get("caveats") or []),
        heatmap=heatmap,
    )


def screen(
    clinical: dict | str | Path, *, mri: str | Path | None = None,
    explain: bool = True, out_dir: str | Path | None = None,
) -> Differential:
    """Run the dementia differential on one patient (blocking).

    Parameters
    ----------
    clinical:
        Raw UDS variables as a dict (e.g. ``{"NACCAGE": 78, "NACCMMSE": 22,
        "SEX": 2, "NACCNE4S": 1}``) or a path to a ``.json``/``.csv`` record. Keys
        must be raw UDS names; unmapped keys are ignored and surfaced as a caveat.
    mri:
        Optional path to the patient's precomputed SwinUNETR embedding
        (``(1, 768, 4, 4, 4)`` ``.npy``). When given, the embedding is fed to the
        model and the differential becomes multimodal (``imaging_used=True``);
        ``None`` falls back to a clinical-features-only prediction.
    explain:
        When True, also compute the feature × label attribution heatmap (one extra
        forward pass per feature, plus an MRI row when ``mri`` is set); disable for
        the lowest-latency path.
    """
    tool = _get_tool()
    raw = tool.analyze(clinical=clinical, mri=mri, explain=explain, out_dir=out_dir)
    return _to_result(raw)


async def screen_async(
    clinical: dict | str | Path, *, mri: str | Path | None = None,
    explain: bool = True, out_dir: str | Path | None = None,
) -> Differential:
    """Async wrapper over :func:`screen` that keeps the event loop free."""
    return await asyncio.to_thread(
        screen, clinical, mri=mri, explain=explain, out_dir=out_dir)
