"""Pydantic data contracts shared across the CRYCHIC pipeline.

`schemas.py` is the single source of truth for I/O (CLAUDE.md §5): the MCP tool
signatures and the agent code both import from here, so a change to a contract
updates both sides together.

One validated model per pipeline step, in execution order (CLAUDE.md §3):

    XueFeatures        S1  note/dict → Xue feature dict + per-feature confidence
    Differential       S2  xue_predict — the 13-label differential, CLINICAL ONLY
    ImagingPlan        S3  router — which imaging checks to dispatch (the one decision)
    Metric             S4b/c  one derived quantitative biomarker (value+threshold+ref)
    FindingCard        S4d  the radiology-style card a clinician reviews
    Reconciliation     S5  per-etiology concordance of probs vs imaging
    UnifiedEvidence    S5  the merged, provenance-tagged evidence bundle
    CrychicReport      S6  the signed-off-able Markdown draft + self-check verdict
    CaseEvidence       what get_case_evidence returns (a selectable view)

Two invariants are encoded structurally here, not just by convention:

* **Numbers are computed, never authored (Inv #2).** Every quantitative value
  lives on a :class:`Metric` and travels with the ``threshold`` it is judged
  against and the ``reference`` it comes from. The sentence generator (S4d) only
  *injects* those digits as fixed tokens — it cannot invent a number, because the
  number does not originate in prose.
* **No fabricated measurements (Inv #2, #7).** A biomarker is either ``measured``,
  ``unavailable`` (no weights/segmentation), or an explicit ``abstain`` (no
  structural correlate for that etiology). There is no synthetic-number path.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# --- The Xue 2024 label set (13 non-mutually-exclusive probabilities). --------
# Cognitive stage is one axis; etiology is a second, independent axis — a patient
# can carry elevated P(AD), P(VD) and P(MCI) at once (real-world comorbidity).
STAGE_LABELS: tuple[str, ...] = ("NC", "MCI", "DE")
ETIOLOGY_LABELS: tuple[str, ...] = (
    "AD", "LBD", "VD", "PRD", "FTD", "NPH", "SEF", "PSY", "TBI", "ODE",
)
ALL_LABELS: tuple[str, ...] = STAGE_LABELS + ETIOLOGY_LABELS

# Etiologies with no defended off-the-shelf structural MONAI metric (CLAUDE.md
# §2.8, §4). For these we output "no imaging correlate available — rely on clinical
# features" and NEVER imply imaging confirmed or cleared them. FTD is here after the
# v0.5 trim: its lobar-Z check used a synthetic (non-atlas-matched) norm and was
# retired, so FTD is now left to the clinical features. ODE maps to a BraTS tumor
# read in principle, but that axis is niche and not built here, so it also abstains.
ABSTAIN_ETIOLOGIES: tuple[str, ...] = ("PRD", "SEF", "PSY", "TBI", "LBD", "FTD")


class Modality(str, Enum):
    """An imaging input that may or may not be present for a case."""

    T1 = "t1"        # structural T1w — free whole-brain segmentation
    FLAIR = "flair"  # FLAIR — needed for the WMH/Fazekas (VD) axis


class ImagingCheck(str, Enum):
    """A structural biomarker the router may dispatch (CLAUDE.md §4).

    Each check speaks to exactly one etiology and needs exactly one modality. The
    full per-check declaration (thresholds, references, overlay, sentences) lives in
    :mod:`crychic.checks`; this enum and the two maps below are the minimal
    schema-level identity the router reads to gate on modality.
    """

    HIPPO_Z = "hippo_z"   # AD  — hippocampal w-score (TIV-normalized) [T1]
    EVANS = "evans"       # NPH — automated Evans-like screening flag  [T1]
    FAZEKAS = "fazekas"   # VD  — WMH volume → Fazekas grade           [FLAIR]


# check → (etiology it informs, modality it requires). The router reads these so
# it never dispatches a check whose modality is absent (CLAUDE.md §3, §4).
CHECK_ETIOLOGY: dict[ImagingCheck, str] = {
    ImagingCheck.HIPPO_Z: "AD",
    ImagingCheck.EVANS: "NPH",
    ImagingCheck.FAZEKAS: "VD",
}
CHECK_MODALITY: dict[ImagingCheck, Modality] = {
    ImagingCheck.HIPPO_Z: Modality.T1,
    ImagingCheck.EVANS: Modality.T1,
    ImagingCheck.FAZEKAS: Modality.FLAIR,
}


class MetricStatus(str, Enum):
    """Whether a biomarker was actually measured — guards against fake numbers."""

    MEASURED = "measured"          # a real model/geometry forward pass produced the value
    UNAVAILABLE = "unavailable"    # weights/segmentation/modality missing — no value
    ABSTAIN = "abstain"            # no structural correlate exists for this etiology


# ============================================================================ #
# S1 — feature extraction
# ============================================================================ #

class XueFeatures(BaseModel):
    """S1 output: the structured UDS features that feed xue_predict.

    ``source`` records how the features were obtained: ``structured`` when the
    caller already passed a UDS dict (the demo path — confidences are 1.0), or
    ``extracted`` when an LLM parsed them out of a free-text clinical note.
    """

    features: dict[str, Any] = Field(
        default_factory=dict, description="Raw UDS variables fed to the model."
    )
    confidences: dict[str, float] = Field(
        default_factory=dict,
        description="Per-feature extraction confidence in [0,1]; 1.0 when structured.",
    )
    source: str = Field("structured", description="'structured' | 'extracted'.")
    caveats: list[str] = Field(default_factory=list)


# ============================================================================ #
# S2 — the dementia differential (Xue 2024, multimodal: clinical + MRI embedding)
# ============================================================================ #

class AttributionHeatmap(BaseModel):
    """Feature × 13-label leave-one-out occlusion attribution.

    ``values[i][j]`` = P(full)[label_j] − P(without feature_i)[label_j]; a
    positive value means feature ``i`` pushed label ``j`` up. When the MRI embedding
    was fed to the model there is also an "MRI (img)" row = P(with MRI) −
    P(clinical-only).
    """

    kind: str
    note: str
    rows: list[str]
    columns: list[str]
    values: list[list[float]]
    csv_path: str | None = None
    png_path: str | None = None


class Differential(BaseModel):
    """S2 output: the 13-label dementia differential from the Xue 2024 model.

    Carries the two probability axes separately plus the flat 13-label map, and
    exposes the handful of probabilities the router reads by name. ``imaging_used``
    records whether the SwinUNETR MRI embedding was fed to the model: when True the
    differential is multimodal, so S5 reconciliation against the structural finding
    cards is a *consistency check on already-imaging-informed probabilities*, not an
    independent second opinion — the report wording reflects that.
    """

    stage_probs: dict[str, float] = Field(
        ..., description="P over cognitive stage: NC / MCI / DE."
    )
    etiology_probs: dict[str, float] = Field(
        ..., description="P over the 10 etiologies (AD, LBD, VD, ...)."
    )
    all_probs: dict[str, float] = Field(..., description="Flat map over all 13 labels.")
    stage_top: str
    etiology_top: str

    imaging_used: bool = Field(
        False, description="True when the MRI embedding was fed to the Xue model."
    )
    n_clinical_features: int = Field(
        0, description="How many raw UDS variables actually mapped into the model."
    )
    input_id: str | None = None
    caveats: list[str] = Field(default_factory=list)
    heatmap: AttributionHeatmap | None = None

    # --- router conveniences: the rules in S3 read these by name. -------------
    def p(self, label: str) -> float:
        """Probability for any label, 0.0 if absent (never raises on a typo)."""
        return float(self.all_probs.get(label, 0.0))

    @property
    def p_ad(self) -> float:
        return self.p("AD")

    @property
    def p_mci(self) -> float:
        return self.p("MCI")

    @property
    def p_vd(self) -> float:
        return self.p("VD")

    @property
    def impaired(self) -> bool:
        """True when the most likely cognitive stage is MCI or dementia."""
        return self.stage_top in ("MCI", "DE")


# ============================================================================ #
# S3 — the imaging plan (the one real LLM decision — Inv #5)
# ============================================================================ #

class ImagingPlan(BaseModel):
    """S3 output: which imaging checks to dispatch, why, and what we abstain on.

    The LLM router *chooses* ``checks`` from those whose modality is present;
    ``fired_rules`` is the deterministic audit trail (and the offline fallback's
    output), and ``abstained`` names the etiologies we explicitly will not speak
    to from imaging (Inv #8).
    """

    checks: list[ImagingCheck] = Field(default_factory=list)
    rationale: str = Field("", description="LLM clinical reasoning for the selection.")
    fired_rules: list[str] = Field(
        default_factory=list,
        description="Human-readable rule firings, e.g. 'P(AD)=0.62 ≥ 0.30 → hippo_z'.",
    )
    abstained: list[str] = Field(
        default_factory=list,
        description="Etiologies with no imaging correlate (Inv #8) or no modality.",
    )


# ============================================================================ #
# S4 — derived biomarkers and the cards that present them
# ============================================================================ #

class Metric(BaseModel):
    """One derived quantitative biomarker (S4b/S4c).

    Every measured metric travels with the ``threshold`` it is judged against,
    the ``comparator`` that defines abnormality, and the ``reference`` it comes
    from — so a finding can always be traced and the number never floats free of
    its meaning (Inv #2, #7).
    """

    etiology: str = Field(..., description="Which differential label this informs.")
    name: str = Field(..., description="Display name, e.g. 'Hippocampal volume Z'.")
    value: float | None = Field(None, description="The computed value; None if unmeasured.")
    unit: str = ""
    threshold: float = Field(..., description="Abnormality cutoff for `comparator`.")
    comparator: str = Field("<", description="One of '<', '<=', '>', '>=' vs threshold.")
    abnormal: bool = Field(False, description="True when value crosses the threshold.")
    status: MetricStatus = MetricStatus.MEASURED
    reference: str = Field("", description="Citation / normative source for the threshold.")
    caveats: list[str] = Field(default_factory=list)

    @property
    def direction(self) -> str:
        """Qualitative direction vs threshold: 'below' | 'above' | 'within' | 'n/a'."""
        if self.value is None:
            return "n/a"
        if self.value < self.threshold:
            return "below"
        if self.value > self.threshold:
            return "above"
        return "within"


class KeySlice(BaseModel):
    """The slice a clinician should verify a finding on."""

    plane: str = Field(..., description="'axial' | 'coronal' | 'sagittal'.")
    index: int = Field(..., description="Slice index along that plane.")


class FindingCard(BaseModel):
    """S4d output: a radiology-style finding card (CLAUDE.md §1, §7).

    The credibility anchor of the UI. ``sentence`` is generated with the metric's
    digits injected as **fixed tokens** (Inv #2) — the language model may shape
    the qualitative framing but can never emit a different number. A negative
    finding still produces a card, and an abstain produces a card too (Inv #7/#8).
    """

    etiology: str
    title: str
    metric: Metric | None = None
    sentence: str = Field("", description="Guardrailed impression; digits are injected tokens.")
    polarity: str = Field(
        "negative", description="'supporting' | 'negative' | 'abstain'."
    )
    key_slice: KeySlice | None = None
    overlay_png_path: str | None = None
    references: list[str] = Field(default_factory=list)


# ============================================================================ #
# S5 — reconciliation, conflicts, and the unified bundle (plain code)
# ============================================================================ #

class ReconClass(str, Enum):
    """How a differential label lines up with its imaging axis (CLAUDE.md §3)."""

    CONCORDANT = "concordant"        # high prob + supporting imaging
    DISCORDANT = "discordant"        # high prob + contradicting imaging → flag
    CLINICAL_ONLY = "clinical_only"  # high prob + no imaging axis available
    INCIDENTAL = "incidental"        # low prob + positive imaging


class Reconciliation(BaseModel):
    """One etiology's prob-vs-imaging concordance verdict."""

    etiology: str
    prob: float
    recon: ReconClass
    evidence: list[str] = Field(default_factory=list)


class ConflictSeverity(str, Enum):
    INFO = "info"
    CAUTION = "caution"
    IMPORTANT = "important"


class Conflict(BaseModel):
    """An evidence conflict, surfaced for the clinician — never silently resolved."""

    conflict_id: str
    name: str
    description: str
    severity: ConflictSeverity = ConflictSeverity.CAUTION
    evidence: list[str] = Field(default_factory=list)


class UnifiedEvidence(BaseModel):
    """S5 output: the merged, provenance-tagged evidence bundle the reasoner reads."""

    case_id: str
    differential: Differential
    plan: ImagingPlan
    cards: list[FindingCard] = Field(default_factory=list)
    reconciliations: list[Reconciliation] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    provenance: list[str] = Field(
        default_factory=list,
        description="Tool/reference that produced each value, for auditability.",
    )


# ============================================================================ #
# S6 — the report
# ============================================================================ #

class CrychicReport(BaseModel):
    """The Markdown draft plus the self-check loop's verdict."""

    markdown: str
    self_check_passed: bool = False
    iterations: int = 0
    remaining_violations: list[str] = Field(default_factory=list)
    warning: str | None = Field(
        None, description="Set when the critic loop did not converge within the cap."
    )


# ============================================================================ #
# Aggregate evidence (what get_case_evidence returns)
# ============================================================================ #

class CaseEvidence(BaseModel):
    """Structured intermediate evidence for one case, by field name.

    ``get_case_evidence`` returns whichever of these the caller asked for; all are
    optional because a case is built up stage by stage.
    """

    case_id: str
    features: XueFeatures | None = None
    differential: Differential | None = None
    plan: ImagingPlan | None = None
    cards: list[FindingCard] | None = None
    reconciliations: list[Reconciliation] | None = None
    conflicts: list[Conflict] | None = None
    report: CrychicReport | None = None
