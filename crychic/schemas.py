
"""Pydantic data contracts shared across the CRYCHIC pipeline.

One validated model per pipeline stage, in execution order:

    Tier1Result        Stage 1  clinical screening (Xue 2024)
    RoutingDecision    Stage 2  which imaging tools the rules selected
    CentiloidResult    Stage 3  MYGO amyloid-PET quantification
    AnatomyResult      Stage 3  MONAI wholeBrainSeg structural metrics
    Tier2Result        Stage 3  the imaging results, bundled
    ClinicalPattern    Stage 4  matched AD-spectrum pattern (1 of 6)
    Conflict           Stage 4  surfaced evidence conflict (never overridden)
    CrychicReport      Stage 5  the signed-off-able Markdown draft
    CaseEvidence       what get_case_evidence returns (all of the above)

Keeping every stage's output as an explicit, validated model is what makes the
pipeline auditable (CDS principle #3, traceable claims): each numeric field
travels with the ``threshold`` it is judged against and the ``reference`` it
comes from, so the reasoner can never emit a bare number.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# --- The Xue 2024 label set (13 non-mutually-exclusive probabilities). --------
# Cognitive stage is one axis; etiologies are a second, indepen
# dent axis — a
# patient can have elevated P(AD), P(VD) and P(MCI) at once.
STAGE_LABELS: tuple[str, ...] = ("NC", "MCI", "DE")
ETIOLOGY_LABELS: tuple[str, ...] = (
    "AD", "LBD", "VD", "PRD", "FTD", "NPH", "SEF", "PSY", "TBI", "ODE",
)
ALL_LABELS: tuple[str, ...] = STAGE_LABELS + ETIOLOGY_LABELS


class AttributionHeatmap(BaseModel):
    """Feature x 13-label leave-one-out occlusion attribution.

    ``values[i][j]`` = P(full)[label_j] - P(without feature_i)[label_j]; a
    positive value means feature ``i`` pushed label ``j`` up. Mirrors the
    structure produced by ``adrd_tool``; file paths are populated only when an
    output directory was supplied.
    """

    kind: str
    note: str
    rows: list[str]
    columns: list[str]
    values: list[list[float]]
    csv_path: str | None = None
    png_path: str | None = None


class Tier1Result(BaseModel):
    """Structured output of Stage 1 (clinical screening).

    Carries the two probability axes separately plus the flat 13-label map, and
    exposes the handful of probabilities the router's hard-coded rules read.
    """

    stage_probs: dict[str, float] = Field(
        ..., description="P over cognitive stage: NC / MCI / DE."
    )
    etiology_probs: dict[str, float] = Field(
        ..., description="P over the 10 etiologies (AD, LBD, VD, ...)."
    )
    all_probs: dict[str, float] = Field(
        ..., description="Flat map over all 13 labels."
    )
    stage_top: str
    etiology_top: str

    n_clinical_features: int = Field(
        0, description="How many raw UDS variables actually mapped into the model."
    )
    imaging_used: bool = False
    input_id: str | None = None
    caveats: list[str] = Field(default_factory=list)
    heatmap: AttributionHeatmap | None = None

    # --- router conveniences: the rules in Stage 2 read these by name. --------
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


# --- Stage 2: routing ---------------------------------------------------------
# The LLM writes the rationale; these hard-coded rules decide what actually runs
# (CLAUDE.md §2.3). Splitting the two keeps the execution path auditable.

class ToolName(str, Enum):
    CENTILOID = "centiloid"   # MYGO amyloid-PET quantification
    MONAI = "monai"           # wholeBrainSeg anatomy


class RoutingDecision(BaseModel):
    """Which imaging tools fire, why (rules), and the LLM's prose rationale."""

    selected_tools: list[ToolName]
    fired_rules: list[str] = Field(
        default_factory=list,
        description="Human-readable rule firings, e.g. 'P(AD)=0.62 ≥ 0.30 → centiloid'.",
    )
    rationale: str = Field(
        "", description="LLM-generated clinical reasoning for the selection."
    )


# --- Stage 3: imaging ---------------------------------------------------------
# `source` distinguishes a real model forward pass from a deterministic
# placeholder, so a report can never silently present a stub as a measurement.

class ImagingSource(str, Enum):
    MODEL = "model"           # real checkpoint / bundle forward pass
    SYNTHETIC = "synthetic"   # deterministic placeholder (no weights available)


class CentiloidResult(BaseModel):
    """MYGO-Centiloid: amyloid-PET burden on the standardized Centiloid scale."""

    centiloid: float = Field(..., description="Centiloid value (Klunk 2015 scale).")
    positive: bool = Field(..., description="centiloid ≥ threshold.")
    threshold: float = 20.0
    tracer: str | None = None
    cortical_suvr: float | None = None
    source: ImagingSource = ImagingSource.MODEL
    reference: str = "Klunk et al. 2015 (Centiloid standardization); GAAIN ≥20 positivity."
    caveats: list[str] = Field(default_factory=list)


class AnatomyResult(BaseModel):
    """MONAI wholeBrainSeg: structural metrics derived from a T1 segmentation."""

    hippocampus_left_mm3: float | None = None
    hippocampus_right_mm3: float | None = None
    hippocampus_total_mm3: float | None = None
    hippocampus_zscore: float | None = Field(
        None, description="Hippocampal volume Z vs age/ICV norms; ≤ -1.5 = atrophy."
    )
    ventricle_volume_mm3: float | None = None
    ventricular_index: float | None = Field(
        None, description="Lateral ventricle / brain volume; elevated → ventriculomegaly."
    )
    dominant_atrophy: str = Field(
        "none",
        description="'medial_temporal' | 'frontotemporal' | 'global' | 'ventriculomegaly' | 'none'.",
    )
    n_labels: int = Field(0, description="Segmented structures (133 for the UNEST protocol).")
    atrophy_zscore_threshold: float = -1.5
    source: ImagingSource = ImagingSource.MODEL
    reference: str = "MONAI wholeBrainSeg (UNEST, 133-label protocol); hippocampal Z per NIA-AA structural criteria."
    caveats: list[str] = Field(default_factory=list)


class Tier2Result(BaseModel):
    """The Stage-3 imaging bundle. Each tool is optional (gated by the router)."""

    centiloid: CentiloidResult | None = None
    anatomy: AnatomyResult | None = None

    @property
    def amyloid_positive(self) -> bool | None:
        return None if self.centiloid is None else self.centiloid.positive


# --- Stage 4: aggregation (pattern + conflicts) -------------------------------

class ClinicalPattern(BaseModel):
    """One of 6 predefined AD-spectrum patterns (CLAUDE.md §2.5)."""

    pattern_id: int = Field(..., ge=1, le=6)
    name: str
    rationale: str
    supporting_evidence: list[str] = Field(default_factory=list)
    confidence: str = Field("moderate", description="'low' | 'moderate' | 'high'.")


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


# --- Stage 5: report ----------------------------------------------------------

class CrychicReport(BaseModel):
    """The Markdown draft plus the self-check loop's verdict."""

    markdown: str
    self_check_passed: bool = False
    iterations: int = 0
    remaining_violations: list[str] = Field(default_factory=list)
    warning: str | None = Field(
        None, description="Set when the critic loop did not converge within the cap."
    )


# --- Aggregate evidence (what get_case_evidence returns) ----------------------

class CaseEvidence(BaseModel):
    """Structured intermediate evidence for one case, by field name.

    ``get_case_evidence`` returns whichever of these the caller asked for; all
    are optional because a case is built up stage by stage.
    """

    case_id: str
    tier1: Tier1Result | None = None
    routing: RoutingDecision | None = None
    tier2: Tier2Result | None = None
    pattern: ClinicalPattern | None = None
    conflicts: list[Conflict] | None = None
    report: CrychicReport | None = None
