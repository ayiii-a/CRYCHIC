"""FLAIR white-matter-hyperintensity burden → an approximate Fazekas grade (VD axis).

The vascular-dementia axis (CLAUDE.md §4): WMH burden is measured from a FLAIR
volume, the hyperintensity volume is mapped to an approximate Fazekas-equivalent
grade, and grade ≥ 2 is the vascular-burden flag. No stock MONAI WMH bundle exists,
so by default the burden comes from a deterministic FLAIR-intensity proxy
(:func:`_wmh_volume_ml_intensity`) — the FLAIR is skull-stripped and on the MNI grid,
so robust thresholding gives a coarse but honest volume, caveated like the
"Evans-like" index. A learned bundle is an optional upgrade (``CRYCHIC_WMH_BUNDLE``).

**This axis abstains without FLAIR.** The bundled demo cohort is mostly T1-only, so
in practice the tool returns an ``UNAVAILABLE`` metric with a card that says "no FLAIR
— vascular burden not assessed from imaging; rely on clinical features" (Inv #8 —
never imply imaging confirmed or cleared VD). No FLAIR is ever fabricated and no
WMH number is invented.

Like :mod:`crychic.segmentation`, torch/MONAI imports live inside the methods that
need them, so this module imports without torch; an optional bundle is loaded once at
server startup (:func:`warmup`) and stays resident (Inv #3).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import checks
from .schemas import ImagingCheck, Metric, MetricStatus

_REPO_ROOT = Path(__file__).resolve().parent.parent

# There is no stock WMH bundle in the MONAI catalog under a stable slug (names drift
# across Model-Zoo releases and none resolves against monaihosting), so the VD axis
# does NOT depend on one by default. When FLAIR is present we compute WMH burden with
# a deterministic, self-contained intensity proxy (:func:`_wmh_volume_ml_intensity`) —
# the FLAIR is skull-stripped and on the MNI grid (scripts/coreg_flair.py), so robust
# intensity thresholding gives a coarse but honest hyperintensity volume, framed like
# the "Evans-like" index, never a fabricated grade. A learned model is still supported
# as an upgrade: install a bundle under ``checkpoints/monai_bundles/<name>/`` (configs/
# + models/model.pt) and set CRYCHIC_WMH_BUNDLE (and CRYCHIC_WMH_BUNDLE_SOURCE if it
# must be fetched); then the bundle takes precedence over the intensity proxy.
_WMH_BUNDLE = os.environ.get("CRYCHIC_WMH_BUNDLE", "")  # empty → intensity proxy
_WMH_BUNDLE_SOURCE = os.environ.get("CRYCHIC_WMH_BUNDLE_SOURCE", "monaihosting")

_SPEC = checks.CHECKS[ImagingCheck.FAZEKAS]  # threshold / reference / name / unit
_FAZEKAS_THRESHOLD = int(_SPEC.threshold)


# Cache of the FLAIR volume + WMH mask (same grid) so the finding card's annotated
# key slice can be rendered without recomputing — mirrors the segmentation cache.
@dataclass
class WmhResult:
    image: np.ndarray   # FLAIR intensity volume (H, W, D), float32
    mask: np.ndarray    # WMH mask on the same grid, bool
    vox_mm3: float
    wmh_ml: float


_WMH_CACHE: dict[str, WmhResult] = {}


def get_wmh(flair_path: str) -> "WmhResult | None":
    """The cached WMH result for a FLAIR, or None if not computed yet."""
    return _WMH_CACHE.get(flair_path)


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
        mask = (logits.argmax(dim=1)[0].detach().cpu().numpy() > 0)
        vox_mm3 = 1.0
        if affine is not None:
            a = np.asarray(affine)
            vox_mm3 = float(abs(np.linalg.det(a[:3, :3]))) or 1.0
        wmh_ml = float(mask.sum() * vox_mm3) / 1000.0  # mm³ → mL
        image_np = np.asarray(img[0].detach().cpu().numpy(), dtype=np.float32)
        _WMH_CACHE[flair_path] = WmhResult(image_np, mask, vox_mm3, wmh_ml)
        return wmh_ml

    @classmethod
    def get(cls) -> "_WmhSeg":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance


# ============================================================================ #
# Deterministic intensity proxy (no learned model) + dispatcher
# ============================================================================ #

def _bundle_installed() -> bool:
    """True when a WMH bundle is present on disk (configs/ exists)."""
    if not _WMH_BUNDLE:
        return False
    bundle_dir = Path(os.environ.get(
        "CHECKPOINT_DIR", _REPO_ROOT / "checkpoints")).expanduser() / "monai_bundles"
    return (bundle_dir / _WMH_BUNDLE / "configs").is_dir()


def _wmh_volume_ml_intensity(flair_path: str) -> tuple[float, float, float]:
    """Coarse WMH volume from a skull-stripped FLAIR by robust intensity thresholding.

    Returns ``(wmh_ml, brain_ml, wmh_fraction_pct)``. FLAIR suppresses CSF, so the
    bulk of in-brain intensity is GM/WM and hyperintensities sit in the upper tail:
    flag voxels above ``median + k·MAD`` (a robust, scale-free centre+spread). This is
    NOT a learned segmentation — a screening proxy, caveated on every result.
    """
    import nibabel as nib

    # Canonical orientation so the cached image/mask slice the same way render.take
    # expects (matches the T1 overlay path).
    img = nib.as_closest_canonical(nib.load(flair_path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    affine = np.asarray(img.affine)
    vox_mm3 = float(abs(np.linalg.det(affine[:3, :3]))) or 1.0

    pos = data[data > 0]
    if pos.size == 0:
        _WMH_CACHE[flair_path] = WmhResult(data, np.zeros(data.shape, bool), vox_mm3, 0.0)
        return 0.0, 0.0, 0.0
    # Drop the faint resampling fringe, then take robust brain-tissue statistics.
    med0 = float(np.median(pos))
    brain = data > (0.10 * med0)
    vals = data[brain]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    if mad <= 0:
        mad = float(vals.std()) or 1.0

    k = float(os.environ.get("CRYCHIC_WMH_SD", "3.0"))
    thr = med + k * mad
    wmh = brain & (data > thr)

    brain_ml = float(brain.sum()) * vox_mm3 / 1000.0
    wmh_ml = float(wmh.sum()) * vox_mm3 / 1000.0
    frac = (100.0 * wmh_ml / brain_ml) if brain_ml > 0 else 0.0
    _WMH_CACHE[flair_path] = WmhResult(data, wmh, vox_mm3, wmh_ml)
    return wmh_ml, brain_ml, frac


def _measure_wmh(flair_path: str) -> tuple[float, list[str]]:
    """Return ``(wmh_volume_ml, method_caveats)`` — learned bundle if available, else
    the deterministic intensity proxy. Raises only if the explicitly-requested bundle
    fails (so the caller surfaces an actionable message rather than silently degrading).
    """
    explicit = bool(_WMH_BUNDLE)
    if explicit or _bundle_installed():
        try:
            ml = _WmhSeg.get().wmh_volume_ml(flair_path)
            return ml, [f"WMH volume ≈ {ml:.1f} mL via MONAI bundle '{_WMH_BUNDLE}'.",
                        "Fazekas grade is a volumetric approximation of a visual scale; "
                        "confirm pattern (periventricular vs deep) on FLAIR."]
        except Exception:
            if explicit:          # the user asked for this bundle — don't hide the failure
                raise

    wmh_ml, brain_ml, frac = _wmh_volume_ml_intensity(flair_path)
    caveats = [
        f"WMH volume ≈ {wmh_ml:.1f} mL ({frac:.1f}% of {brain_ml:.0f} mL brain) by a "
        "coarse FLAIR-intensity threshold (robust median + k·MAD), NOT a learned "
        "segmentation — a screening proxy; confirm pattern (periventricular vs deep) "
        "on FLAIR. Set CRYCHIC_WMH_BUNDLE to a WMH model for a learned grade.",
    ]
    if frac > 5.0:
        caveats.append("Flagged fraction is high — check FLAIR intensity scaling / "
                       "registration before trusting the magnitude.")
    return wmh_ml, caveats


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
        wmh_ml, method_caveats = _measure_wmh(flair_path)
    except Exception as exc:  # only an explicitly-requested bundle failing reaches here
        return Metric(value=None, abnormal=False, status=MetricStatus.UNAVAILABLE,
                      caveats=[f"FLAIR is present but the requested WMH bundle "
                               f"'{_WMH_BUNDLE}' could not be used — VD axis not "
                               f"assessed ({exc}). Install it under "
                               f"checkpoints/monai_bundles/{_WMH_BUNDLE}/, fix "
                               "CRYCHIC_WMH_BUNDLE / CRYCHIC_WMH_BUNDLE_SOURCE, or "
                               "unset CRYCHIC_WMH_BUNDLE to use the intensity proxy."],
                      **base)

    grade = _fazekas_from_volume_ml(wmh_ml)
    return Metric(value=float(grade), abnormal=grade >= _FAZEKAS_THRESHOLD,
                  status=MetricStatus.MEASURED, caveats=method_caveats, **base)


def warmup() -> None:
    """Pre-load the WMH bundle so the first FLAIR request is inference-only (Inv #3).

    No-op when no bundle is configured/installed — the default VD path is the
    deterministic intensity proxy, which has nothing to warm up.
    """
    if _WMH_BUNDLE or _bundle_installed():
        _WmhSeg.get()
