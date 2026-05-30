"""FLAIR white-matter-hyperintensity burden → an approximate Fazekas grade (VD axis).

The vascular-dementia axis (CLAUDE.md §4): a MONAI WMH segmentation bundle runs on
a FLAIR volume, the hyperintensity volume is mapped to an approximate Fazekas-
equivalent grade, and grade ≥ 2 is the vascular-burden flag.

**This axis abstains without FLAIR.** The bundled demo cohort is T1-only, so in
practice the tool returns an ``UNAVAILABLE`` metric with a card that says "no FLAIR
— vascular burden not assessed from imaging; rely on clinical features" (Inv #8 —
never imply imaging confirmed or cleared VD). No FLAIR is ever fabricated and no
WMH number is invented.

Like :mod:`crychic.segmentation`, torch/MONAI imports live inside the methods that
need them, so this module imports without torch; the bundle is loaded once at
server startup (:func:`warmup`) and stays resident (Inv #3).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from . import checks
from .schemas import ImagingCheck, Metric, MetricStatus

_REPO_ROOT = Path(__file__).resolve().parent.parent

# NOTE (§7-style TODO): there is no stock WMH bundle in the MONAI catalog under a
# stable slug — names drift across Model-Zoo releases and the default below does NOT
# resolve against monaihosting. To enable the VD axis, install a WMH bundle locally
# under ``checkpoints/monai_bundles/<name>/`` (configs/ + models/model.pt) and point
# CRYCHIC_WMH_BUNDLE at it, or set CRYCHIC_WMH_BUNDLE_SOURCE to a source/slug that
# resolves (e.g. a "github" model-zoo entry). Until then fazekas() abstains with an
# explicit "model not installed" caveat — never a fabricated grade.
_WMH_BUNDLE = os.environ.get("CRYCHIC_WMH_BUNDLE", "wmh_segmentation")
_WMH_BUNDLE_SOURCE = os.environ.get("CRYCHIC_WMH_BUNDLE_SOURCE", "monaihosting")

_SPEC = checks.CHECKS[ImagingCheck.FAZEKAS]  # threshold / reference / name / unit
_FAZEKAS_THRESHOLD = int(_SPEC.threshold)


def _fazekas_from_volume_ml(wmh_ml: float) -> int:
    """Map total WMH volume (mL) to an approximate Fazekas-equivalent grade 0–3.

    Volumetric proxy for a visual scale — caveated on every result. Bins follow
    the usual minimal / punctate / early-confluent / confluent progression.
    """
    if wmh_ml < 1.5:
        return 0
    if wmh_ml < 6.0:
        return 1
    if wmh_ml < 15.0:
        return 2
    return 3


# ============================================================================ #
# Warm singleton around the MONAI WMH bundle
# ============================================================================ #

class _WmhSeg:
    _instance: "_WmhSeg | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        import torch

        from monai.bundle import ConfigParser, download

        self.device = os.environ.get(
            "TIER2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        bundle_dir = Path(os.environ.get(
            "CHECKPOINT_DIR", _REPO_ROOT / "checkpoints")).expanduser() / "monai_bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        self._root = bundle_dir / _WMH_BUNDLE
        if not (self._root / "configs").is_dir():
            download(name=_WMH_BUNDLE, bundle_dir=str(bundle_dir), source=_WMH_BUNDLE_SOURCE)

        import sys
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))
        parser = ConfigParser()
        cfg = self._root / "configs" / "inference.json"
        parser.read_config(str(cfg if cfg.exists()
                               else self._root / "configs" / "inference.yaml"))
        parser["bundle_root"] = str(self._root)
        self.preprocessing = parser.get_parsed_content("preprocessing", instantiate=True)
        self.inferer = parser.get_parsed_content("inferer", instantiate=True)
        net = parser.get_parsed_content("network_def", instantiate=True).to(self.device)
        weights = self._root / "models" / "model.pt"
        state = torch.load(str(weights), map_location=self.device, weights_only=False)
        state = state.get("model", state) if isinstance(state, dict) else state
        net.load_state_dict(state)
        self.network = net.eval()
        self._torch = torch

    def wmh_volume_ml(self, flair_path: str) -> float:
        torch = self._torch
        batch = self.preprocessing({"image": flair_path})
        img = batch["image"]
        affine = getattr(img, "affine", None)
        x = img.unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            logits = self.inferer(x, self.network)
        mask = logits.argmax(dim=1)[0].detach().cpu().numpy()
        vox_mm3 = 1.0
        if affine is not None:
            a = np.asarray(affine)
            vox_mm3 = float(abs(np.linalg.det(a[:3, :3]))) or 1.0
        return float((mask > 0).sum() * vox_mm3) / 1000.0  # mm³ → mL

    @classmethod
    def get(cls) -> "_WmhSeg":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance


# ============================================================================ #
# Public
# ============================================================================ #

def fazekas(flair_path: str | None) -> Metric:
    """WMH→Fazekas Metric for the VD axis; abstains (no value) without FLAIR.

    A measured grade ≥ 2 flags vascular burden. When FLAIR is absent or the model
    is unavailable, the metric is ``UNAVAILABLE`` with no fabricated value —
    surfaced downstream as an explicit "no imaging assessment" card (Inv #8).
    """
    base = dict(etiology=_SPEC.etiology, name=_SPEC.metric_name,
                threshold=_SPEC.threshold, comparator=_SPEC.comparator,
                unit=_SPEC.unit, reference=_SPEC.reference)

    if not flair_path:
        return Metric(value=None, abnormal=False, status=MetricStatus.UNAVAILABLE,
                      caveats=["No FLAIR supplied — vascular burden is not assessed "
                               "from imaging; rely on clinical features for a "
                               "vascular contribution."], **base)

    try:
        wmh_ml = _WmhSeg.get().wmh_volume_ml(flair_path)
    except Exception as exc:  # missing bundle/weights/torch — no fabricated grade
        # Distinct from the no-FLAIR branch above: FLAIR *was* supplied; only the
        # segmentation model is missing. Say so explicitly so this never reads as
        # "no imaging" and stays actionable (which bundle, how to install it).
        return Metric(value=None, abnormal=False, status=MetricStatus.UNAVAILABLE,
                      caveats=[f"FLAIR is present but the WMH segmentation model "
                               f"'{_WMH_BUNDLE}' is not installed — VD axis not "
                               f"assessed ({exc}). Install a WMH bundle under "
                               f"checkpoints/monai_bundles/{_WMH_BUNDLE}/ or set "
                               "CRYCHIC_WMH_BUNDLE / CRYCHIC_WMH_BUNDLE_SOURCE to a "
                               "bundle that resolves."],
                      **base)

    grade = _fazekas_from_volume_ml(wmh_ml)
    return Metric(value=float(grade), abnormal=grade >= _FAZEKAS_THRESHOLD,
                  status=MetricStatus.MEASURED,
                  caveats=[f"WMH volume ≈ {wmh_ml:.1f} mL (whole-brain).",
                           "Fazekas grade is a volumetric approximation of a visual "
                           "scale; confirm pattern (periventricular vs deep) on FLAIR."],
                  **base)


def warmup() -> None:
    """Pre-load the WMH bundle so the first FLAIR request is inference-only (Inv #3)."""
    _WmhSeg.get()
