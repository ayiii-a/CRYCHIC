"""Deterministic structural geometry — the project's credibility anchor (CLAUDE.md §9).

Every quantitative biomarker in a CRYCHIC report originates here or in another
tool's compute path, NEVER in prose (Inv #2). These are pure functions: a label
volume + the bundle's channel map go in, numbers come out. No model forward pass,
no LLM, no I/O — which makes them unit-testable on synthetic arrays without torch
or MONAI.

What lives here (CLAUDE.md §4):
    hippocampus_z   AD  — hippocampal w-score (residualized for age, sex, TIV)
    evans_like_index NPH — automated Evans-like index (see the honesty caveat)
    pick_key_slice  the slice a clinician should verify a finding on

The frontotemporal lobar-Z check was retired in v0.5: its normative frontotemporal
fraction was synthetic rather than atlas-matched, so it could not be defended
clinically. FTD now abstains and is left to the clinical features (Inv #8).

Two honesty constraints, stated up front because they bound what the numbers mean:

* **Label IDs are matched by NAME, not by hardcoded index (CLAUDE.md §7).** The
  UNEST 133-label IDs are unverified, so we select structures by case-insensitive
  whole-token matching against the bundle's ``channel_def`` (e.g. "hippocampus"
  matches "Left-Hippocampus" but not "Right-PHG---parahippocampal-gyrus"). If the
  bundle ships no ``channel_def``, the metrics degrade to ``None`` with a caveat
  rather than guessing.
* **Evans is approximated (CLAUDE.md §7).** A brain-only segmentation has no skull,
  so the true inner-table denominator is unavailable; we use the maximal
  intracranial *brain* width instead. The result is labeled an "automated
  Evans-like index" and is biased low versus a guideline-grade Evans — it must
  never be presented as the exact radiological measurement.

Orientation assumption: the MONAI-preprocessed grid is treated as RAS-oriented
(axis 0 = R↔L, axis 1 = P↔A, axis 2 = I↔S), matching the slice viewer. If a future
bundle emits a different orientation, transverse widths and the axial slice picker
would need the axes remapped — this is called out as a caveat on the metric.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# RAS axis convention (see module docstring).
PLANE_AXIS: dict[str, int] = {"sagittal": 0, "coronal": 1, "axial": 2}
_AX_LR, _AX_PA, _AX_IS = 0, 1, 2

# Thresholds (kept named so the report can cite them; mirror CLAUDE.md §4).
HIPPO_Z_THRESHOLD = -1.5
EVANS_THRESHOLD = 0.30

# Hippocampal w-score model (the correct dementia normalization — see hippocampus_z):
#   V_pred = b0 + b_age·age + b_sex_male·sex_male + b_tiv·TIV ;  w = (V - V_pred) / sd
# Coefficients load from norms/hippo_wscore.json; these defaults are an UNCALIBRATED
# placeholder until fit on a same-pipeline cognitively-normal cohort
# (scripts/fit_hippo_wscore.py). Every use is flagged in the resulting caveats.
_WSCORE_PATH = Path(__file__).resolve().parent / "norms" / "hippo_wscore.json"
_MIN_VALID_NORM_N = 20  # below this, the norm stays flagged as not-yet-validated
_DEFAULT_WSCORE = {
    "intercept": 5125.0, "b_age": -35.0, "b_sex_male": 560.0, "b_tiv": 0.004,
    "residual_sd": 700.0, "age_mean": 65.0, "sex_male_mean": 0.45,
    "tiv_mean": 1_200_000.0, "n": 0, "source": "placeholder",
}


# ============================================================================ #
# Label selection by channel name (never by hardcoded UNEST id — §7)
# ============================================================================ #

_TOKEN_SPLIT = re.compile(r"[-_\s]+")


def label_tokens(name: str) -> set[str]:
    return {t for t in _TOKEN_SPLIT.split(name.lower()) if t}


def label_indices(channel_def: dict[int, str], *needles: str) -> list[int]:
    """Indices whose channel name contains *every* needle as a whole token.

    AND semantics across needles: ``("lateral", "ventricle")`` matches
    "Right-Lateral-Ventricle" but not "3rd-Ventricle"; ``("hippocampus",)``
    matches "Left-Hippocampus" but not "...parahippocampal...".
    """
    needles_l = {n.lower() for n in needles}
    return [idx for idx, name in channel_def.items()
            if needles_l.issubset(label_tokens(name))]


def _side_indices(channel_def: dict[int, str], idxs: list[int], side: str) -> list[int]:
    """Subset of ``idxs`` whose channel name is on the given side ('left'/'right')."""
    out = []
    for i in idxs:
        nm = channel_def.get(i, "").lower()
        if side in nm or nm.startswith(side[0] + " "):
            out.append(i)
    return out


def volume_mm3(labels: np.ndarray, idxs: list[int], vox_mm3: float) -> float:
    """Total volume of the given label indices, in mm³."""
    if not idxs:
        return 0.0
    return float(np.isin(labels, idxs).sum() * vox_mm3)


def total_intracranial_volume(labels: np.ndarray, vox_mm3: float) -> float:
    """ICV proxy = total segmented brain tissue (all non-zero labels), in mm³.

    A brain-only segmentation has no skull, so this is brain tissue volume, not
    true intracranial volume — close enough for TIV-normalization, and caveated
    wherever it is used.
    """
    return float((labels > 0).sum() * vox_mm3)


# ============================================================================ #
# Hippocampal Z (AD) — TIV-normalized, vs age/sex norms
# ============================================================================ #

@dataclass
class ZResult:
    z: float | None
    raw_mm3: float | None
    tiv_mm3: float | None
    used_fallback_norm: bool = False
    caveats: list[str] = field(default_factory=list)


def _load_wscore() -> tuple[dict, bool]:
    """Return (coefficients, used_default).

    Reads norms/hippo_wscore.json (merged over the built-in defaults so a partial
    file still works); falls back to the placeholder defaults with used_default=True
    when the file is absent or unreadable.
    """
    if not _WSCORE_PATH.exists():
        return dict(_DEFAULT_WSCORE), True
    try:
        import json
        return {**_DEFAULT_WSCORE, **json.loads(_WSCORE_PATH.read_text())}, False
    except (OSError, ValueError):
        return dict(_DEFAULT_WSCORE), True


def hippocampus_z(
    hippo_total_mm3: float | None,
    tiv_mm3: float | None,
    *,
    age: float | None = None,
    sex: str | None = None,
) -> ZResult:
    """Hippocampal w-score: a residualized Z adjusted for age, sex, and TIV.

    The correct normalization for a dementia hippocampal biomarker (CLAUDE.md §4).
    Instead of a proportional ratio against age/sex bins, the expected volume is a
    regression on age, sex and TIV fit in cognitively-normal references, and

        w = (V_observed - V_predicted) / residual_SD,    atrophy at w < -1.5

    mirroring the BrainChart (Bethlehem 2022) / Potvin 2016 normative-modelling
    approach. Coefficients come from ``norms/hippo_wscore.json``; the shipped
    defaults are an UNCALIBRATED placeholder (flagged in the caveats) until fit on a
    cohort segmented by THIS bundle (scripts/fit_hippo_wscore.py). Missing age/sex
    are mean-imputed from the model.
    """
    if not hippo_total_mm3 or not tiv_mm3:
        return ZResult(z=None, raw_mm3=hippo_total_mm3, tiv_mm3=tiv_mm3,
                       caveats=["Hippocampus or TIV unavailable; w-score not computed."])

    c, used_default = _load_wscore()
    sd = float(c.get("residual_sd") or 0.0)
    if sd <= 0:
        return ZResult(z=None, raw_mm3=round(hippo_total_mm3, 1),
                       tiv_mm3=round(tiv_mm3, 1), used_fallback_norm=True,
                       caveats=["Normative residual SD missing; w-score not computed."])

    sex_male = {"M": 1.0, "F": 0.0}.get((sex or "").upper())
    age_imputed, sex_imputed = age is None, sex_male is None
    age_use = float(age) if age is not None else float(c.get("age_mean", 65.0))
    sex_use = sex_male if sex_male is not None else float(c.get("sex_male_mean", 0.45))

    v_pred = (float(c["intercept"]) + float(c["b_age"]) * age_use
              + float(c["b_sex_male"]) * sex_use + float(c["b_tiv"]) * tiv_mm3)
    w = round((hippo_total_mm3 - v_pred) / sd, 2)

    n = int(float(c.get("n") or 0))
    placeholder = used_default or c.get("source") != "cohort" or n < _MIN_VALID_NORM_N
    caveats = []
    if placeholder:
        caveats.append("Hippocampal w-score uses UNCALIBRATED placeholder coefficients "
                       "(not fit on a cognitively-normal cohort segmented by this "
                       "bundle) — treat the absolute value as indicative only; "
                       "calibrate with scripts/fit_hippo_wscore.py.")
    else:
        caveats.append(f"Hippocampal w-score from coefficients fit on {n} "
                       "cognitively-normal subjects segmented by this bundle "
                       "(age/sex/TIV-adjusted).")
    if age_imputed or sex_imputed:
        miss = " and ".join(x for x, m in (("age", age_imputed), ("sex", sex_imputed)) if m)
        caveats.append(f"Missing {miss}; mean-imputed from the normative model.")
    caveats.append("TIV here is total segmented brain tissue (ICV proxy), not a "
                   "skull-bounded intracranial volume.")
    return ZResult(z=w, raw_mm3=round(hippo_total_mm3, 1), tiv_mm3=round(tiv_mm3, 1),
                   used_fallback_norm=used_default, caveats=caveats)


def hippocampus_z_from_seg(
    labels: np.ndarray, channel_def: dict[int, str], vox_mm3: float,
    *, age: float | None = None, sex: str | None = None,
) -> ZResult:
    """Convenience: locate the hippocampus by name, then :func:`hippocampus_z`."""
    if not channel_def:
        return ZResult(None, None, None,
                       caveats=["Bundle metadata lacked a channel_def; hippocampus "
                                "could not be located by name."])
    hippo_idx = label_indices(channel_def, "hippocampus")
    hippo_total = volume_mm3(labels, hippo_idx, vox_mm3) or None
    tiv = total_intracranial_volume(labels, vox_mm3) or None
    return hippocampus_z(hippo_total, tiv, age=age, sex=sex)


# ============================================================================ #
# Evans-like index (NPH) — APPROXIMATED (no skull; §7)
# ============================================================================ #

@dataclass
class EvansResult:
    index: float | None
    axial_index: int | None
    frontal_horn_width_vox: int | None = None
    intracranial_width_vox: int | None = None
    caveats: list[str] = field(default_factory=list)


def evans_like_index(
    labels: np.ndarray, channel_def: dict[int, str], vox_mm3: float | None = None,
) -> EvansResult:
    """Automated Evans-like index = frontal-horn span ÷ intracranial brain width.

    Approximates the radiological Evans index: the numerator is the maximal
    transverse (R↔L) span of the lateral-ventricle frontal horns, and the
    denominator is the maximal transverse span of the brain on the SAME axial
    slice — used because a brain-only segmentation has no inner skull table.
    Reported as an "automated Evans-like index" (biased low vs guideline Evans).
    Ventriculomegaly threshold: index > 0.30.
    """
    caveat = ("Automated Evans-like index: the denominator is intracranial brain "
              "width (no skull in a brain-only segmentation), not the inner skull "
              "table — biased low vs a guideline-grade Evans; verify on the image.")
    if not channel_def:
        return EvansResult(None, None, caveats=[
            "Bundle metadata lacked a channel_def; ventricles could not be located."])

    vent_idx = label_indices(channel_def, "lateral", "ventricle")
    vent = np.isin(labels, vent_idx) if vent_idx else np.zeros(labels.shape, bool)
    brain = labels > 0
    if not vent.any() or not brain.any():
        return EvansResult(None, None, caveats=[caveat,
                           "No lateral-ventricle or brain voxels found."])

    # Anterior half (frontal horns sit anteriorly; RAS → larger P-A index = anterior).
    pa = np.where(brain.any(axis=(_AX_LR, _AX_IS)))[0]
    pa_mid = int((pa.min() + pa.max()) / 2)
    anterior = np.zeros(labels.shape, bool)
    anterior_slices = [slice(None)] * 3
    anterior_slices[_AX_PA] = slice(pa_mid, None)
    anterior[tuple(anterior_slices)] = True
    vent_ant = vent & anterior

    best_idx, best_num = None, 0
    for z in range(labels.shape[_AX_IS]):
        sl = [slice(None)] * 3
        sl[_AX_IS] = z
        vrow = vent_ant[tuple(sl)]
        if not vrow.any():
            continue
        xs = np.where(vrow.any(axis=1 if _AX_LR == 0 else 0))[0]
        width = int(xs.max() - xs.min() + 1)
        if width > best_num:
            best_num, best_idx = width, z

    if best_idx is None:
        return EvansResult(None, None, caveats=[caveat,
                           "No anterior lateral-ventricle voxels on any axial slice."])

    sl = [slice(None)] * 3
    sl[_AX_IS] = best_idx
    brow = brain[tuple(sl)]
    bxs = np.where(brow.any(axis=1 if _AX_LR == 0 else 0))[0]
    denom = int(bxs.max() - bxs.min() + 1)
    if denom <= 0:
        return EvansResult(None, best_idx, best_num, None, caveats=[caveat,
                           "Intracranial width was zero on the selected slice."])

    return EvansResult(index=round(best_num / denom, 3), axial_index=best_idx,
                       frontal_horn_width_vox=best_num, intracranial_width_vox=denom,
                       caveats=[caveat])


# ============================================================================ #
# Key-slice picker — the slice the clinician should verify a finding on
# ============================================================================ #

def pick_key_slice(mask: np.ndarray, plane: str) -> int:
    """Index along ``plane`` holding the most region voxels, so the view centers
    on the structure being analyzed. Falls back to the mid-slice for an empty mask.
    """
    axis = PLANE_AXIS[plane]
    if mask.any():
        other = tuple(i for i in range(mask.ndim) if i != axis)
        return int(mask.sum(axis=other).argmax())
    return mask.shape[axis] // 2
