"""Stage 3 — Tier-2 imaging tools.

Three independent models, run in parallel by the pipeline:

* **MONAI wholeBrainSeg** (``run_anatomy``) — *real* inference. Downloads the
  MONAI Model-Zoo whole-brain parcellation bundle on first use, runs a
  sliding-window forward pass over the T1, and derives hippocampal volume +
  Z-score, lateral-ventricle volume + ventricular index, and a dominant-atrophy
  call from the label map. Loaded once and kept warm.
* **MYGO-Centiloid** (``run_centiloid``) — amyloid-PET quantification. No
  trained checkpoint and no PET in the demo cohort, so this emits a
  **deterministic synthetic** value (seeded by the case + nudged by the Tier-1
  amyloid prior) and flags ``source=synthetic``.
* **MUJICA** (``run_epvs``) — enlarged-perivascular-space burden. Same
  deterministic-synthetic treatment, nudged by the Tier-1 vascular prior.

Every tool has a blocking implementation and an ``*_async`` wrapper that offloads
to a worker thread so the event loop stays free. A synthetic result is never
silently presented as a measurement — ``ImagingSource`` records which is which,
and the pipeline turns a tool failure into a caveat rather than a crash.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import (
    AnatomyResult,
    CentiloidResult,
    EPVSResult,
    ImagingSource,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================ #
# Shared helpers
# ============================================================================ #

def _seed_from(*parts: Any) -> int:
    """Stable 32-bit seed from arbitrary inputs (so a case is reproducible)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def _checkpoint_dir() -> Path:
    return Path(os.environ.get("CHECKPOINT_DIR", _REPO_ROOT / "checkpoints")).expanduser()


# ============================================================================ #
# Segmentation cache — the preprocessed volume + label map, for region overlays
# ============================================================================ #
# MONAI's preprocessing resamples the T1, so the label map lives on a different
# grid than the raw .nii.gz. We stash the *preprocessed* image alongside its
# labels (same grid) so the web layer can overlay regions in correct alignment
# without re-running inference. Keyed by t1_path (one volume per demo case).

@dataclass
class SegResult:
    image: np.ndarray            # preprocessed intensity volume (H, W, D), float32
    labels: np.ndarray           # parcellation on the same grid (H, W, D), int16
    vox_mm3: float
    channel_def: dict[int, str] = field(default_factory=dict)
    hippo_idx: list[int] = field(default_factory=list)
    vent_idx: list[int] = field(default_factory=list)


_SEG_CACHE: dict[str, SegResult] = {}


def get_segmentation(t1_path: str) -> SegResult | None:
    """The cached preprocessed image + label map for a T1, or None if MONAI has
    not run on it yet (used by the web UI to draw region overlays)."""
    return _SEG_CACHE.get(t1_path)


# ============================================================================ #
# MONAI wholeBrainSeg — real inference
# ============================================================================ #

# Normative total (L+R) hippocampal volume for the Z-score. This is a documented
# internal reference, NOT an age/ICV-matched norm — the report must caveat it.
_HIPPO_NORM_MEAN_MM3 = 6500.0
_HIPPO_NORM_SD_MM3 = 800.0
# Lateral-ventricle / hippocampus volume ratio above which we flag
# ventriculomegaly out of proportion to medial-temporal atrophy.
_VENTRICULOMEGALY_RATIO = 6.0

_BUNDLE_NAME = os.environ.get(
    "CRYCHIC_MONAI_BUNDLE", "wholeBrainSeg_Large_UNEST_segmentation"
)


class _MonaiWholeBrainSeg:
    """Lazy, warm singleton around the MONAI parcellation bundle.

    The bundle ships its own preprocessing/inferer/network, so we drive it via
    ``ConfigParser`` rather than re-deriving transforms. The label→structure map
    is read from the bundle metadata, which keeps the volume extraction robust
    to the exact 133-label protocol.
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

    # -- construction ------------------------------------------------------- #
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
            f"None of {names} found under {self._bundle_root / 'configs'}"
        )

    def _load(self, torch) -> None:
        import sys

        from monai.bundle import ConfigParser

        # The bundle's network_def may reference custom modules under scripts/.
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

    # -- inference ---------------------------------------------------------- #
    def segment(self, t1_path: str) -> tuple[np.ndarray, float, np.ndarray]:
        """Return (label volume, per-voxel mm³, preprocessed intensity volume).

        The intensity volume is the channel-0 image the network actually saw,
        on the *same* (resampled) grid as ``labels`` — so the two align voxel for
        voxel for region overlays.
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


def _label_indices(channel_def: dict[int, str], *needles: str) -> list[int]:
    needles_l = [n.lower() for n in needles]
    return [idx for idx, name in channel_def.items()
            if any(n in name.lower() for n in needles_l)]


def _zscore(value: float, mean: float, sd: float) -> float:
    return round((value - mean) / sd, 2) if sd else 0.0


def run_anatomy(t1_path: str) -> AnatomyResult:
    """Real MONAI wholeBrainSeg on one T1 → structural metrics."""
    seg = _MonaiWholeBrainSeg.get()
    labels, vox_mm3, image_np = seg.segment(t1_path)
    chan = seg.channel_def

    def volume(idxs: list[int]) -> float:
        if not idxs:
            return 0.0
        return float(np.isin(labels, idxs).sum() * vox_mm3)

    hippo_l_idx = _label_indices(chan, "hippocampus")  # may merge L/R if names lack side
    left_idx = [i for i in hippo_l_idx if "left" in chan[i].lower() or chan[i].lower().startswith("l ")]
    right_idx = [i for i in hippo_l_idx if "right" in chan[i].lower() or chan[i].lower().startswith("r ")]
    vent_idx = _label_indices(chan, "lateral ventricle", "ventricle")

    # Stash the aligned image + labels so the web layer can overlay regions.
    _SEG_CACHE[t1_path] = SegResult(
        image=image_np, labels=labels, vox_mm3=vox_mm3, channel_def=chan,
        hippo_idx=hippo_l_idx, vent_idx=vent_idx,
    )

    hippo_left = volume(left_idx) or None
    hippo_right = volume(right_idx) or None
    hippo_total = volume(hippo_l_idx) or None
    vent_vol = volume(vent_idx) or None

    caveats: list[str] = []
    if not chan:
        caveats.append("Bundle metadata lacked a channel_def; structure volumes "
                       "could not be extracted by name.")
    hippo_z = (_zscore(hippo_total, _HIPPO_NORM_MEAN_MM3, _HIPPO_NORM_SD_MM3)
               if hippo_total else None)
    if hippo_z is not None:
        caveats.append("Hippocampal Z uses an internal normative mean/SD, not an "
                       "age/sex/ICV-matched atlas; treat as approximate.")

    # Ventricular index ≈ ventricle / total segmented brain volume.
    brain_vol = float((labels > 0).sum() * vox_mm3)
    vent_index = round(vent_vol / brain_vol, 4) if (vent_vol and brain_vol) else None

    atrophy = _classify_atrophy(hippo_z, hippo_total, vent_vol)

    return AnatomyResult(
        hippocampus_left_mm3=hippo_left,
        hippocampus_right_mm3=hippo_right,
        hippocampus_total_mm3=hippo_total,
        hippocampus_zscore=hippo_z,
        ventricle_volume_mm3=vent_vol,
        ventricular_index=vent_index,
        dominant_atrophy=atrophy,
        n_labels=int(len(np.unique(labels))),
        source=ImagingSource.MODEL,
        caveats=caveats,
    )


def _classify_atrophy(
    hippo_z: float | None, hippo_total: float | None, vent_vol: float | None
) -> str:
    ventriculomegaly = (
        hippo_total and vent_vol and vent_vol > _VENTRICULOMEGALY_RATIO * hippo_total
    )
    if hippo_z is not None and hippo_z <= -1.5:
        return "medial_temporal"
    if ventriculomegaly:
        return "ventriculomegaly"
    return "none"


async def run_anatomy_async(t1_path: str) -> AnatomyResult:
    return await asyncio.to_thread(run_anatomy, t1_path)


# ============================================================================ #
# MYGO-Centiloid — deterministic synthetic
# ============================================================================ #

def run_centiloid(
    pet_path: str | None,
    tracer: str | None,
    *,
    amyloid_prior: float = 0.5,
    seed_key: str = "",
) -> CentiloidResult:
    """Synthetic amyloid-PET burden, biased by the Tier-1 amyloid prior.

    ``amyloid_prior`` (≈ max(P(AD), P(MCI))) sets the probability the case lands
    amyloid-positive, so the demo stays internally coherent. Deterministic given
    ``seed_key`` + ``pet_path``.
    """
    rng = np.random.RandomState(_seed_from("centiloid", seed_key, pet_path))
    p = float(min(max(amyloid_prior, 0.0), 1.0))
    threshold = 20.0
    if rng.random_sample() < p:                       # positive
        centiloid = float(round(rng.uniform(28, 110), 1))
    else:                                             # negative
        centiloid = float(round(rng.uniform(-8, 16), 1))
    suvr = round(1.0 + max(centiloid, 0) / 100.0 * 1.2, 2)

    caveats = [
        f"Synthetic placeholder — no MYGO-Centiloid checkpoint present; value is "
        f"deterministic (seed-derived, nudged by P≈{p:.2f}), not measured.",
    ]
    if pet_path is None:
        caveats.append("No amyloid PET was supplied; amyloid status is inferred, "
                       "not quantified from a scan.")

    return CentiloidResult(
        centiloid=centiloid,
        positive=centiloid >= threshold,
        threshold=threshold,
        tracer=tracer,
        cortical_suvr=suvr,
        source=ImagingSource.SYNTHETIC,
        caveats=caveats,
    )


async def run_centiloid_async(
    pet_path: str | None, tracer: str | None, *, amyloid_prior: float = 0.5,
    seed_key: str = "",
) -> CentiloidResult:
    return await asyncio.to_thread(
        run_centiloid, pet_path, tracer, amyloid_prior=amyloid_prior, seed_key=seed_key
    )


# ============================================================================ #
# MUJICA EPVS — deterministic synthetic
# ============================================================================ #

def run_epvs(
    t1_path: str | None,
    *,
    vascular_prior: float = 0.3,
    caa_prior: float = 0.3,
    seed_key: str = "",
) -> EPVSResult:
    """Synthetic EPVS burden, biased by the Tier-1 vascular / CAA priors.

    Higher ``vascular_prior`` raises total burden; higher ``caa_prior`` shifts
    the distribution toward centrum-semiovale predominance (the CAA surrogate).
    Deterministic given ``seed_key`` + ``t1_path``.
    """
    rng = np.random.RandomState(_seed_from("mujica", seed_key, t1_path))
    v = float(min(max(vascular_prior, 0.0), 1.0))
    c = float(min(max(caa_prior, 0.0), 1.0))

    total = float(round(rng.uniform(150, 400) + v * rng.uniform(300, 1400), 1))
    cso_frac = float(min(0.85, max(0.15, rng.uniform(0.3, 0.55) + 0.35 * c)))
    cso = round(total * cso_frac, 1)
    bg = round(total - cso, 1)

    if cso_frac >= 0.6:
        distribution = "CSO-predominant"
    elif cso_frac <= 0.4:
        distribution = "BG-predominant"
    else:
        distribution = "mixed"

    # Ordinal 0–4 burden from total volume, scaled by the vascular prior.
    grade = int(np.clip(round(total / 350.0), 0, 4))

    return EPVSResult(
        total_volume_mm3=total,
        bg_volume_mm3=bg,
        cso_volume_mm3=cso,
        distribution=distribution,
        burden_grade=grade,
        source=ImagingSource.SYNTHETIC,
        caveats=[
            "Synthetic placeholder — no MUJICA checkpoint present; volumes are "
            "deterministic (seed-derived), not segmented from a scan.",
        ],
    )


async def run_epvs_async(
    t1_path: str | None, *, vascular_prior: float = 0.3, caa_prior: float = 0.3,
    seed_key: str = "",
) -> EPVSResult:
    return await asyncio.to_thread(
        run_epvs, t1_path, vascular_prior=vascular_prior, caa_prior=caa_prior,
        seed_key=seed_key,
    )


def warmup_monai() -> None:
    """Pre-load the MONAI bundle so the first real request is fast (demo pre-cache)."""
    _MonaiWholeBrainSeg.get()
