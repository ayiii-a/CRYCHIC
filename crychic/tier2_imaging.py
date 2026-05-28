
"""Stage 3 — Tier-2 imaging tools.

Two independent models, run in parallel by the pipeline:

* **MONAI wholeBrainSeg** (``run_anatomy``) — *real* inference. Downloads the
  MONAI Model-Zoo whole-brain parcellation bundle on first use, runs a
  sliding-window forward pass over the T1, and derives hippocampal volume +
  Z-score, lateral-ventricle volume + ventricular index, and a dominant-atrophy
  call from the label map. Loaded once and kept warm.
* **MYGO-Centiloid** (``run_centiloid``) — *real* inference when a checkpoint
  is available (set ``MYGO_CHECKPOINT`` or drop the file at
  ``<checkpoint_dir>/mygo_centiloid_best.pt``); falls back to a deterministic
  synthetic value (seeded by the case + nudged by the Tier-1 amyloid prior,
  flagged ``source=synthetic``) when the checkpoint, package, or input is not
  usable. ``.npy`` shaped ``(1, 128, 128, 128)`` is the preferred input —
  ``.nii.gz`` is accepted as a best-effort fallback (MONAI resample + min-max).

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
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import (
    AnatomyResult,
    CentiloidResult,
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


_TOKEN_SPLIT = re.compile(r"[-_\s]+")


def _label_tokens(name: str) -> set[str]:
    return {t for t in _TOKEN_SPLIT.split(name.lower()) if t}


def _label_indices(channel_def: dict[int, str], *needles: str) -> list[int]:
    """Indices whose channel name contains *every* needle as a whole token.

    Tokens are split on hyphens, underscores, and whitespace and matched
    case-insensitively, so ``"hippocampus"`` matches ``"Left-Hippocampus"``
    but NOT ``"Right-PHG---parahippocampal-gyrus"``. With multiple needles,
    AND semantics — ``("lateral", "ventricle")`` matches
    ``"Right-Lateral-Ventricle"`` but not ``"3rd-Ventricle"``.
    """
    needles_l = {n.lower() for n in needles}
    return [idx for idx, name in channel_def.items()
            if needles_l.issubset(_label_tokens(name))]


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
    vent_idx = _label_indices(chan, "lateral", "ventricle")

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
# MYGO-Centiloid — real inference (warm singleton) with synthetic fallback
# ============================================================================ #
# Model expects ``(1, 128, 128, 128)`` float32 in ``[0, 1]``, conditioned on a
# tracer id from the checkpoint's ``tracer_map``. ``.npy`` in that exact shape
# is preferred (it matches training preprocessing); ``.nii.gz`` is accepted as
# a best-effort fallback (MONAI resample + ScaleIntensity), with a caveat
# surfaced on the result so the report cannot conceal the preprocessing drift.

_MYGO_INPUT_HW = 128
_MYGO_CKPT_ENV = "MYGO_CHECKPOINT"
_MYGO_DEFAULT_CKPT = "mygo_centiloid_best.pt"
_CENTILOID_POS_THRESHOLD = 20.0

# The repo ships a vendored, locally-patched copy of the MYGO-Centiloid source
# under ``vendor/MYGO-Centiloid/`` — see vendor/MYGO-Centiloid/mygo_centiloid/__init__.py.
# Putting it first on sys.path means the MCP is independent of whether the
# upstream package is pip-installed (and whether that install is well-formed).
_MYGO_VENDOR_DIR = _REPO_ROOT / "vendor" / "MYGO-Centiloid"


def _mygo_checkpoint_path() -> Path:
    explicit = os.environ.get(_MYGO_CKPT_ENV)
    if explicit:
        return Path(explicit).expanduser()
    return _checkpoint_dir() / _MYGO_DEFAULT_CKPT


def _ensure_vendored_mygo_on_path() -> None:
    """Insert the vendored MYGO source dir at ``sys.path[0]`` if it exists.

    If an unpatched ``mygo_centiloid`` was already imported (and cached in
    ``sys.modules``), evict it so the next import picks the vendored copy.
    """
    import sys

    if not (_MYGO_VENDOR_DIR / "mygo_centiloid" / "__init__.py").exists():
        return
    vendor_str = str(_MYGO_VENDOR_DIR)
    if sys.path and sys.path[0] == vendor_str:
        return
    sys.path.insert(0, vendor_str)
    # Drop any prior (broken) cached mygo_centiloid so the next import re-resolves.
    for name in [m for m in sys.modules if m == "mygo_centiloid"
                 or m.startswith("mygo_centiloid.")]:
        del sys.modules[name]


class _MygoCentiloid:
    """Lazy, warm singleton around the MYGO-Centiloid amyloid-PET regressor.

    Construction loads the checkpoint and rebuilds the architecture the
    checkpoint was trained with (``ckpt["model"]`` ∈
    ``petresnet`` | ``petresnet_no_film`` | ``petresnet_no_head_emb``). The
    ``tracer_map`` lives in the checkpoint, so the singleton is the source of
    truth for which tracers are accepted at inference time.
    """

    _instance: "_MygoCentiloid | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        import torch

        # Prefer the vendored source under ``vendor/MYGO-Centiloid/`` over any
        # pip-installed copy (the upstream public-review init imports ablation
        # arms that aren't shipped, so a fresh pip install is broken).
        _ensure_vendored_mygo_on_path()

        # The mygo_centiloid package is optional; let ImportError bubble so the
        # caller can fall back to the synthetic path cleanly.
        from mygo_centiloid import PETResNet, PETResNetNoFiLM
        try:
            from mygo_centiloid import PETResNetNoHeadEmb
        except ImportError:  # older releases without this ablation arm
            PETResNetNoHeadEmb = None  # type: ignore[assignment]

        ckpt_path = _mygo_checkpoint_path()
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"MYGO-Centiloid checkpoint not found at {ckpt_path}; set "
                f"${_MYGO_CKPT_ENV} or drop the file there."
            )

        self.device = os.environ.get(
            "TIER2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)

        self.tracer_map: dict[str, int] = dict(ckpt["tracer_map"])
        num_tracers = int(ckpt["num_tracers"])
        model_name = ckpt.get("model", "petresnet")
        emb_dim = int(ckpt.get("emb_dim", 32))
        d_hi = float(ckpt.get("dropout_high", 0.4))
        d_lo = float(ckpt.get("dropout_low", 0.2))

        if model_name == "petresnet_no_film":
            net = PETResNetNoFiLM(num_tracers=num_tracers,
                                  dropout_high=d_hi, dropout_low=d_lo)
        elif model_name == "petresnet_no_head_emb" and PETResNetNoHeadEmb is not None:
            net = PETResNetNoHeadEmb(num_tracers=num_tracers, emb_dim=emb_dim,
                                     dropout_high=d_hi, dropout_low=d_lo)
        else:
            net = PETResNet(num_tracers=num_tracers, emb_dim=emb_dim,
                            dropout_high=d_hi, dropout_low=d_lo)

        # Strip torch.compile prefix if present (matches dev/predict.py).
        state = {k.replace("_orig_mod.", ""): v
                 for k, v in ckpt["model_state_dict"].items()}
        net.load_state_dict(state)
        self.network = net.to(self.device).eval()
        self.model_name = model_name
        self._torch = torch

    def tracer_id(self, tracer: str | None) -> int:
        if tracer is None:
            raise KeyError(
                f"tracer is required (one of {sorted(self.tracer_map)}); got None"
            )
        key = tracer.strip().upper()
        if key not in self.tracer_map:
            raise KeyError(
                f"tracer {tracer!r} not in trained tracer_map "
                f"({sorted(self.tracer_map)})"
            )
        return int(self.tracer_map[key])

    def predict(self, volume: np.ndarray, tracer: str | None) -> float:
        """Forward one preprocessed ``(1, 128, 128, 128)`` volume → centiloid."""
        torch = self._torch
        tid = self.tracer_id(tracer)
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"

        x = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))
        x = x.unsqueeze(0).to(self.device)                # (1, 1, 128, 128, 128)
        t = torch.tensor([tid], dtype=torch.long, device=self.device)
        with torch.no_grad(), torch.amp.autocast(
            device_type=device_type, enabled=(device_type == "cuda")
        ):
            pred = self.network(x, t)
        return float(pred.detach().float().cpu().view(-1)[0].item())

    @classmethod
    def get(cls) -> "_MygoCentiloid":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance


def _load_pet_volume(pet_path: str) -> tuple[np.ndarray, list[str]]:
    """Return ``((1, 128, 128, 128) float32, caveats)`` for a PET path.

    ``.npy`` is consumed as-is (must already be the trained shape). ``.nii.gz``
    is resampled + min-max normalized via MONAI as a best-effort fallback and
    returns a caveat noting that the preprocessing is not training-identical.
    """
    p = Path(pet_path)
    name = p.name.lower()

    if name.endswith(".npy"):
        arr = np.load(str(p))
        if arr.ndim == 3:
            arr = arr[None, ...]
        expected = (1, _MYGO_INPUT_HW, _MYGO_INPUT_HW, _MYGO_INPUT_HW)
        if arr.shape != expected:
            raise ValueError(
                f"Expected .npy of shape {expected}; got {arr.shape}"
            )
        return arr.astype(np.float32, copy=False), []

    if name.endswith(".nii") or name.endswith(".nii.gz"):
        from monai.transforms import (
            Compose, EnsureChannelFirst, LoadImage, Resize, ScaleIntensity,
        )
        tx = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Resize(spatial_size=(_MYGO_INPUT_HW,) * 3, mode="trilinear",
                   align_corners=False),
            ScaleIntensity(minv=0.0, maxv=1.0),
        ])
        vol = np.asarray(tx(str(p)), dtype=np.float32)
        return vol, [
            "PET volume was preprocessed from a raw .nii.gz (MONAI resample + "
            "min-max); training used a fixed pipeline — feed a (1,128,128,128) "
            ".npy for exact reproducibility.",
        ]

    raise ValueError(f"Unsupported PET extension for {pet_path!r}; "
                     "pass .npy or .nii[.gz]")


def run_centiloid(
    pet_path: str | None,
    tracer: str | None,
    *,
    amyloid_prior: float = 0.5,
    seed_key: str = "",
) -> CentiloidResult:
    """Real MYGO-Centiloid inference; falls back to synthetic when unavailable.

    Synthetic fires when (1) no PET path is supplied, (2) the checkpoint or
    ``mygo_centiloid`` package is missing, or (3) the input fails to load or
    the tracer is not in the trained ``tracer_map``. Every fallback flags
    ``source=synthetic`` and surfaces the reason as a caveat — no synthetic
    value can be silently presented as a measurement (CDS principle #3).
    """
    if pet_path is None:
        return _run_centiloid_synthetic(
            pet_path, tracer, amyloid_prior=amyloid_prior, seed_key=seed_key,
            reason="No amyloid PET was supplied; amyloid status is inferred, "
                   "not quantified from a scan.",
        )

    try:
        model = _MygoCentiloid.get()
        volume, load_caveats = _load_pet_volume(pet_path)
        centiloid = round(model.predict(volume, tracer), 1)
    except Exception as exc:
        return _run_centiloid_synthetic(
            pet_path, tracer, amyloid_prior=amyloid_prior, seed_key=seed_key,
            reason=f"MYGO-Centiloid unavailable — fell back to synthetic ({exc}).",
        )

    suvr = round(1.0 + max(centiloid, 0.0) / 100.0 * 1.2, 2)
    return CentiloidResult(
        centiloid=centiloid,
        positive=centiloid >= _CENTILOID_POS_THRESHOLD,
        threshold=_CENTILOID_POS_THRESHOLD,
        tracer=tracer,
        cortical_suvr=suvr,
        source=ImagingSource.MODEL,
        reference=("MYGO-Centiloid (Jia et al. 2026; multitracer-conditioned 3D "
                   "ResNet18, MedAI Spring 2026; val MAE 11.73 CL, r 0.936); "
                   "Klunk et al. 2015 Centiloid scale; GAAIN ≥20 positivity."),
        caveats=load_caveats,
    )


def _run_centiloid_synthetic(
    pet_path: str | None,
    tracer: str | None,
    *,
    amyloid_prior: float,
    seed_key: str,
    reason: str,
) -> CentiloidResult:
    """Deterministic placeholder, used only when MYGO inference is unavailable."""
    rng = np.random.RandomState(_seed_from("centiloid", seed_key, pet_path))
    p = float(min(max(amyloid_prior, 0.0), 1.0))
    if rng.random_sample() < p:                       # positive
        centiloid = float(round(rng.uniform(28, 110), 1))
    else:                                             # negative
        centiloid = float(round(rng.uniform(-8, 16), 1))
    suvr = round(1.0 + max(centiloid, 0) / 100.0 * 1.2, 2)
    return CentiloidResult(
        centiloid=centiloid,
        positive=centiloid >= _CENTILOID_POS_THRESHOLD,
        threshold=_CENTILOID_POS_THRESHOLD,
        tracer=tracer,
        cortical_suvr=suvr,
        source=ImagingSource.SYNTHETIC,
        caveats=[
            f"Synthetic placeholder — value is deterministic (seed-derived, "
            f"nudged by P≈{p:.2f}), not measured. Reason: {reason}",
        ],
    )


async def run_centiloid_async(
    pet_path: str | None, tracer: str | None, *, amyloid_prior: float = 0.5,
    seed_key: str = "",
) -> CentiloidResult:
    return await asyncio.to_thread(
        run_centiloid, pet_path, tracer, amyloid_prior=amyloid_prior, seed_key=seed_key
    )


def warmup_monai() -> None:
    """Pre-load the MONAI bundle so the first real request is fast (demo pre-cache)."""
    _MonaiWholeBrainSeg.get()
