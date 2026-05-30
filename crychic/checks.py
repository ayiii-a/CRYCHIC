"""Single source of truth for the tier-2 structural imaging checks.

Every fact about a structural check that used to be duplicated across
``geometry`` / ``segmentation`` / ``overlay`` / ``agent.reasoner`` now lives on one
:class:`CheckSpec` in the :data:`CHECKS` table: the etiology it informs, the
modality it needs, the abnormality threshold + comparator, the citable reference,
the overlay plane/region/colour, and the finding-card sentence templates. Add or
retire a check by editing THIS table alone — the consumers read it, they do not
re-declare it.

After the v0.5 clinical trim (CLAUDE.md §2.8) only the two defensible T1 checks
survive — the hippocampal z-score (AD) and the Evans-like *screening flag* (NPH) —
plus the FLAIR Fazekas (VD) axis. The frontotemporal lobar-Z check was dropped
because its normative fraction was synthetic rather than atlas-matched, so it could
not be defended at the bedside; FTD now abstains and is left to the clinical
features (Inv #8).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import geometry, render
from .schemas import ImagingCheck, Modality


@dataclass(frozen=True)
class CheckSpec:
    """Everything one structural check needs, in one place.

    ``sentence_abnormal`` / ``sentence_normal`` are ``str.format`` templates whose
    only fields are the metric's own tokens — ``{v}`` (value), ``{thr}``
    (threshold), ``{cmp}`` (comparator) and ``{unit}`` (a leading-space unit or
    "") — so a finding sentence can never contain a number the metric did not
    carry (Inv #2).

    ``plane`` is ``None`` for a check with no T1 overlay (e.g. the FLAIR Fazekas
    axis); ``region_attr`` names the :class:`~crychic.segmentation.SegResult`
    attribute holding the label indices the overlay highlights.
    """

    check: ImagingCheck
    etiology: str
    modality: Modality
    metric_name: str
    unit: str
    threshold: float
    comparator: str
    reference: str
    sentence_abnormal: str
    sentence_normal: str
    # overlay (all None/"" when there is no T1 key-slice for this check)
    plane: str | None = None
    overlay_title: str = ""
    region_attr: str | None = None
    overlay_rgb: tuple[float, float, float] | None = None
    overlay_alpha: float = 0.5
    overlay_label: str = ""

    def sentence(self, value, threshold, comparator, unit: str, *, abnormal: bool) -> str:
        """Render the finding sentence, injecting only the metric's own digits."""
        tmpl = self.sentence_abnormal if abnormal else self.sentence_normal
        return tmpl.format(v=value, thr=threshold, cmp=comparator, unit=unit)


_HIPPO_REF = ("MONAI wholeBrainSeg (UNEST, 133-label); hippocampal z-score "
              "(residualized for age, sex and TIV) per the BrainChart / Potvin 2016 "
              "normative-modelling approach; atrophy at z < -1.5 (NIA-AA "
              "medial-temporal criteria).")
_EVANS_REF = ("MONAI wholeBrainSeg (UNEST) + geometry; automated Evans-like index "
              "(after Evans 1942), a coarse screening proxy with an intracranial "
              "brain-width denominator — not a diagnostic Evans measurement.")
_VD_REF = ("MONAI WMH segmentation (FLAIR); Fazekas grade approximated from total "
           "WMH volume (Fazekas et al. 1987 is a visual scale — this is a "
           "volumetric proxy). Wardlaw STRIVE small-vessel-disease standards.")


CHECKS: dict[ImagingCheck, CheckSpec] = {
    ImagingCheck.HIPPO_Z: CheckSpec(
        check=ImagingCheck.HIPPO_Z,
        etiology="AD",
        modality=Modality.T1,
        metric_name="Hippocampal volume z-score (age/sex/TIV-adjusted)",
        unit="SD",
        threshold=geometry.HIPPO_Z_THRESHOLD,
        comparator="<",
        reference=_HIPPO_REF,
        sentence_abnormal=(
            "Hippocampal volume is reduced — z-score {v}{unit} ({cmp} {thr} = "
            "atrophy) — consistent with medial-temporal neurodegeneration; "
            "correlate clinically."),
        sentence_normal=(
            "Hippocampal volume is within normal limits — z-score {v}{unit} "
            "(atrophy {cmp} {thr}) — no structural support for medial-temporal "
            "neurodegeneration on this measure."),
        plane="coronal",
        overlay_title="Hippocampal level",
        region_attr="hippo_idx",
        overlay_rgb=render.HIPPO_RGB,
        overlay_alpha=0.55,
        overlay_label="hippocampus",
    ),
    ImagingCheck.EVANS: CheckSpec(
        check=ImagingCheck.EVANS,
        etiology="NPH",
        modality=Modality.T1,
        metric_name="Automated Evans-like index (screening flag)",
        unit="",
        threshold=geometry.EVANS_THRESHOLD,
        comparator=">",
        reference=_EVANS_REF,
        sentence_abnormal=(
            "Automated Evans-like index {v} ({cmp} {thr}) — a coarse automated "
            "screening flag for ventricular enlargement that can accompany NPH; it "
            "is not a diagnostic Evans index — assess gait, continence and cognition "
            "and confirm on the image."),
        sentence_normal=(
            "Automated Evans-like index {v} (NPH screening flag {cmp} {thr}) — below "
            "the screening threshold; no ventriculomegaly by this coarse automated "
            "proxy."),
        plane="axial",
        overlay_title="Ventricular level",
        region_attr="vent_idx",
        overlay_rgb=render.VENT_RGB,
        overlay_alpha=0.50,
        overlay_label="lateral ventricles",
    ),
    ImagingCheck.FAZEKAS: CheckSpec(
        check=ImagingCheck.FAZEKAS,
        etiology="VD",
        modality=Modality.FLAIR,
        metric_name="WMH burden → Fazekas grade",
        unit="grade",
        threshold=2.0,
        comparator=">=",
        reference=_VD_REF,
        sentence_abnormal=(
            "White-matter-hyperintensity burden corresponds to an approximate "
            "Fazekas grade {v} ({cmp} {thr}) — supportive of a vascular "
            "contribution; confirm the pattern on FLAIR."),
        sentence_normal=(
            "Approximate Fazekas grade {v} ({cmp} {thr}) — limited "
            "white-matter-hyperintensity burden by this measure."),
        # FLAIR finding — no T1 key-slice overlay.
    ),
}

# Convenience views the consumers read.
CHECKS_BY_ETIOLOGY: dict[str, CheckSpec] = {s.etiology: s for s in CHECKS.values()}
