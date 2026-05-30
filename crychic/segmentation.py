"""MONAI wholeBrainSeg — the structural T1 segmentation, run at most ONCE per case.

This module owns the single whole-brain parcellation that feeds every T1 metric
(hippocampus, ventricles, lobes) — CLAUDE.md Inv #6: one segmentation per case,
never per-metric, because the marginal cost of each additional T1 metric is ~zero.

Two public surfaces, both consumed by ``servers/imaging_server.py`` and by the
in-process pipeline:

    segment(t1_path)                 → run the bundle once, cache the result, summarize
    derive_metric(t1_path, check, …) → read the cache, compute one Metric (geometry only)

``segment`` is the only place that touches torch/MONAI; ``derive_metric`` is pure
numpy + :mod:`crychic.geometry`, so it is unit-testable offline by seeding the
cache with a synthetic :class:`SegResult`. Heavy imports live inside the methods
that need them, so this module imports cleanly without torch installed (the model
is loaded once at server *startup* via :func:`warmup`, then stays resident —
Inv #3 — and a tool call is inference only).
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import checks, geometry
from .schemas import ImagingCheck, Metric, MetricStatus, Modality

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _checkpoint_dir() -> Path:
    return Path(os.environ.get("CHECKPOINT_DIR", _REPO_ROOT / "checkpoints")).expanduser()


_BUNDLE_NAME = os.environ.get(
    "CRYCHIC_MONAI_BUNDLE", "wholeBrainSeg_Large_UNEST_segmentation"
)


# ============================================================================ #
# Segmentation cache — preprocessed image + label map on the SAME grid
# ============================================================================ #
# MONAI's preprocessing resamples the T1, so the label map lives on a different
# grid than the raw .nii.gz. We stash the *preprocessed* image alongside its
# labels (same grid) so overlays align voxel-for-voxel without re-running
# inference. Keyed by t1_path → one segmentation per case (Inv #6).

@dataclass
class SegResult:
    image: np.ndarray            # preprocessed intensity volume (H, W, D), float32
    labels: np.ndarray           # parcellation on the same grid (H, W, D), int16
    vox_mm3: float
    channel_def: dict[int, str] = field(default_factory=dict)
    tiv_mm3: float = 0.0
    hippo_idx: list[int] = field(default_factory=list)
    vent_idx: list[int] = field(default_factory=list)


_SEG_CACHE: dict[str, SegResult] = {}


def get_segmentation(t1_path: str) -> SegResult | None:
    """The cached segmentation for a T1, or None if it has not been run yet."""
    return _SEG_CACHE.get(t1_path)


def put_segmentation(t1_path: str, seg: SegResult) -> None:
    """Seed the cache directly — used by tests to inject a synthetic segmentation."""
    _SEG_CACHE[t1_path] = seg


# ============================================================================ #
# Warm singleton around the MONAI parcellation bundle
# ============================================================================ #

class _MonaiWholeBrainSeg:
    """Lazy, warm singleton — load once, stay resident (Inv #3).

    The bundle ships its own preprocessing/inferer/network, so we drive it via
    ``ConfigParser`` rather than re-deriving transforms. The label→structure map
    is read from the bundle metadata, keeping volume extraction robust to the exact
    133-label protocol (label IDs themselves are unverified — §7 — so we match by
    name downstream).
    """

    _instance: "_MonaiWholeBrainSeg | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        import torch

        self.device = os.environ.get(
            "TIER2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        bundle_dir = _checkpoint_dir() / "monai_bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        self._bundle_root = bundle_dir / _BUNDLE_NAME

        self._download_if_needed(bundle_dir)
        self._load(torch)
        self.channel_def = self._read_channel_def()

    def _download_if_needed(self, bundle_dir: Path) -> None:
        from monai.bundle import download

        if (self._bundle_root / "configs").is_dir():
            return
        download(name=_BUNDLE_NAME, bundle_dir=str(bundle_dir), source="monaihosting")

    def _config_path(self, *names: str) -> Path:
        for n in names:
            p = self._bundle_root / "configs" / n
            if p.exists():
                return p
        raise FileNotFoundError(
            f"None of {names} found under {self._bundle_root / 'configs'}")

    def _load(self, torch) -> None:
        import sys

        from monai.bundle import ConfigParser

        if str(self._bundle_root) not in sys.path:
            sys.path.insert(0, str(self._bundle_root))

        parser = ConfigParser()
        parser.read_config(str(self._config_path("inference.json", "inference.yaml")))
        try:
            parser.read_meta(str(self._config_path("metadata.json")))
        except FileNotFoundError:
            pass
        parser["bundle_root"] = str(self._bundle_root)

        self.preprocessing = parser.get_parsed_content("preprocessing", instantiate=True)
        self.inferer = parser.get_parsed_content("inferer", instantiate=True)
        net = parser.get_parsed_content("network_def", instantiate=True).to(self.device)

        weights = self._bundle_root / "models" / "model.pt"
        state = torch.load(str(weights), map_location=self.device, weights_only=False)
        state = state.get("model", state) if isinstance(state, dict) else state
        net.load_state_dict(state)
        self.network = net.eval()
        self._torch = torch

    def _read_channel_def(self) -> dict[int, str]:
        meta = self._bundle_root / "configs" / "metadata.json"
        if not meta.exists():
            return {}
        data = json.loads(meta.read_text())
        try:
            raw = data["network_data_format"]["outputs"]["pred"]["channel_def"]
            return {int(k): str(v) for k, v in raw.items()}
        except (KeyError, TypeError, ValueError):
            return {}

    def segment(self, t1_path: str) -> tuple[np.ndarray, float, np.ndarray]:
        """Return (label volume, per-voxel mm³, preprocessed intensity volume).

        The intensity volume is the channel-0 image the network actually saw, on
        the same (resampled) grid as ``labels`` — so the two align for overlays.
        """
        torch = self._torch
        batch = self.preprocessing({"image": t1_path})
        img = batch["image"]
        affine = getattr(img, "affine", None)
        x = img.unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            logits = self.inferer(x, self.network)
        labels = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int16)
        image_np = np.asarray(img[0].detach().cpu().numpy(), dtype=np.float32)
        # Restrict the parcellation to the brain. The input is skull-stripped, so
        # NormalizeIntensityd(nonzero=True) leaves the background at exactly 0; the
        # network still argmaxes tissue classes into that background air, which both
        # inflates volumes and lets the key-slice picker land on an empty slice. Drop
        # every label where the image saw no brain.
        labels[image_np == 0] = 0
        return labels, _voxel_volume_mm3(affine), image_np

    @classmethod
    def get(cls) -> "_MonaiWholeBrainSeg":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance


def _voxel_volume_mm3(affine) -> float:
    if affine is None:
        return 1.0
    a = np.asarray(affine)
    return float(abs(np.linalg.det(a[:3, :3]))) or 1.0


# ============================================================================ #
# Public: segment once, then derive metrics from the cache
# ============================================================================ #

def segment(t1_path: str) -> dict:
    """Run wholeBrainSeg once on ``t1_path``, cache the result, return a summary.

    Idempotent per path: a second call returns the cached summary without
    re-running inference (Inv #6). The numeric biomarkers come from
    :func:`derive_metric`, which reads this cache.
    """
    cached = _SEG_CACHE.get(t1_path)
    if cached is None:
        seg_model = _MonaiWholeBrainSeg.get()
        labels, vox_mm3, image_np = seg_model.segment(t1_path)
        chan = seg_model.channel_def
        hippo_idx = geometry.label_indices(chan, "hippocampus")
        vent_idx = geometry.label_indices(chan, "lateral", "ventricle")
        cached = SegResult(
            image=image_np, labels=labels, vox_mm3=vox_mm3, channel_def=chan,
            tiv_mm3=geometry.total_intracranial_volume(labels, vox_mm3),
            hippo_idx=hippo_idx, vent_idx=vent_idx,
        )
        _SEG_CACHE[t1_path] = cached

    return {
        "t1_path": t1_path,
        "n_labels": int(len(np.unique(cached.labels))),
        "tiv_mm3": round(cached.tiv_mm3, 1),
        "hippocampus_total_mm3": round(
            geometry.volume_mm3(cached.labels, cached.hippo_idx, cached.vox_mm3), 1),
        "ventricle_volume_mm3": round(
            geometry.volume_mm3(cached.labels, cached.vent_idx, cached.vox_mm3), 1),
        "has_channel_def": bool(cached.channel_def),
    }


def _unavailable(etiology: str, name: str, threshold: float, comparator: str,
                 reference: str, why: str) -> Metric:
    return Metric(etiology=etiology, name=name, value=None, threshold=threshold,
                  comparator=comparator, abnormal=False,
                  status=MetricStatus.UNAVAILABLE, reference=reference, caveats=[why])


def _crosses(value: float, threshold: float, comparator: str) -> bool:
    """Whether ``value`` is abnormal vs ``threshold`` under ``comparator``."""
    return value < threshold if comparator.startswith("<") else value > threshold


def derive_metric(
    t1_path: str, check: ImagingCheck, *,
    age: float | None = None, sex: str | None = None,
) -> Metric:
    """Compute one structural :class:`Metric` from the cached segmentation.

    Pure geometry — never re-segments (Inv #6). All per-check metadata (name,
    threshold, comparator, reference, unit) comes from :data:`crychic.checks.CHECKS`;
    only the geometry call is dispatched here. Returns an ``UNAVAILABLE`` metric (no
    fabricated value) if the segmentation has not been run for this T1, the check is
    not a T1 check, or the structure could not be located by name.
    """
    spec = checks.CHECKS.get(check)
    if spec is None or spec.modality is not Modality.T1:
        raise ValueError(f"derive_metric does not handle {check!r} (FLAIR check?)")

    meta = dict(etiology=spec.etiology, name=spec.metric_name,
                threshold=spec.threshold, comparator=spec.comparator,
                reference=spec.reference)
    seg = _SEG_CACHE.get(t1_path)
    if seg is None:
        return _unavailable(**meta, why="T1 not yet segmented.")

    if check is ImagingCheck.HIPPO_Z:
        r = geometry.hippocampus_z_from_seg(seg.labels, seg.channel_def, seg.vox_mm3,
                                            age=age, sex=sex)
        value, caveats = r.z, r.caveats
    elif check is ImagingCheck.EVANS:
        r = geometry.evans_like_index(seg.labels, seg.channel_def, seg.vox_mm3)
        value, caveats = r.index, r.caveats
    else:  # registered T1 check with no geometry binding — shouldn't happen
        raise ValueError(f"no geometry bound for {check!r}")

    if value is None:
        return _unavailable(**meta, why="; ".join(caveats))
    return Metric(value=value, unit=spec.unit,
                  abnormal=_crosses(value, spec.threshold, spec.comparator),
                  status=MetricStatus.MEASURED, caveats=caveats, **meta)


def warmup() -> None:
    """Pre-load the bundle so the first real request is inference-only (Inv #3)."""
    _MonaiWholeBrainSeg.get()
