"""S5 — the aggregator (pure Python rules, no LLM).

Merges the clinical differential, the imaging plan, and the finding cards into one
provenance-tagged :class:`~crychic.schemas.UnifiedEvidence` bundle, and does the two
deterministic jobs the reasoner then narrates:

* **Reconciliation** — classify each etiology by how its clinical probability lines
  up with its imaging axis (CLAUDE.md §3):
    high prob + supporting imaging  → concordant
    high prob + contradicting imaging → discordant (flag)
    high prob + no imaging axis      → clinical-only
    low prob  + positive imaging     → incidental
* **Conflicts** — surface structural evidence conflicts (an AD signal without
  medial-temporal atrophy, atrophy without an AD signal, ventriculomegaly without
  an NPH signal, a dementia-level stage with benign structural imaging). Conflicts
  are surfaced, never silently resolved.

These are rules, not a model: the same evidence always yields the same verdicts,
which is what keeps the report auditable. There is deliberately no amyloid logic —
PET is out of scope (CLAUDE.md §8).
"""

from __future__ import annotations

from .schemas import (
    Conflict,
    ConflictSeverity,
    Differential,
    FindingCard,
    ImagingPlan,
    Metric,
    MetricStatus,
    ReconClass,
    Reconciliation,
    UnifiedEvidence,
)

# "Elevated clinical signal" gate per etiology (mirrors the router's gates; the
# abstain etiologies use a generic gate since they have no imaging axis).
_PROB_GATE = {
    "AD": 0.30, "NPH": 0.15, "FTD": 0.30, "VD": 0.20,
    "LBD": 0.30, "PRD": 0.15, "PSY": 0.30, "SEF": 0.30, "TBI": 0.30, "ODE": 0.30,
}
P_AD_STRONG = 0.50


def _cards_by_etiology(cards: list[FindingCard]) -> dict[str, FindingCard]:
    """One card per etiology; a measured card wins over an abstain placeholder."""
    out: dict[str, FindingCard] = {}
    for c in cards:
        prev = out.get(c.etiology)
        if prev is None or (prev.metric is None and c.metric is not None):
            out[c.etiology] = c
    return out


def _measured(metric: Metric | None) -> bool:
    return metric is not None and metric.status is MetricStatus.MEASURED


# ============================================================================ #
# Reconciliation
# ============================================================================ #

def reconcile(diff: Differential, cards: list[FindingCard]) -> list[Reconciliation]:
    """Classify each etiology's clinical-vs-imaging concordance (§3)."""
    by_etio = _cards_by_etiology(cards)
    out: list[Reconciliation] = []

    for etio, gate in _PROB_GATE.items():
        prob = diff.p(etio)
        card = by_etio.get(etio)
        metric = card.metric if card else None
        high = prob >= gate
        measured = _measured(metric)
        abnormal = bool(measured and metric.abnormal)

        if high and abnormal:
            recon = ReconClass.CONCORDANT
            ev = [f"P({etio})={prob:.2f} (≥{gate}) with supporting imaging: "
                  f"{metric.name} {metric.value} ({metric.comparator} {metric.threshold})."]
        elif high and measured and not abnormal:
            recon = ReconClass.DISCORDANT
            ev = [f"P({etio})={prob:.2f} (≥{gate}) but imaging does not support it: "
                  f"{metric.name} {metric.value} (not {metric.comparator} {metric.threshold})."]
        elif high and not measured:
            recon = ReconClass.CLINICAL_ONLY
            ev = [f"P({etio})={prob:.2f} (≥{gate}); no imaging axis available — "
                  "rely on clinical features."]
        elif not high and abnormal:
            recon = ReconClass.INCIDENTAL
            ev = [f"Imaging positive ({metric.name} {metric.value} "
                  f"{metric.comparator} {metric.threshold}) without a strong clinical "
                  f"signal (P({etio})={prob:.2f} <{gate})."]
        else:
            continue  # low prob + normal/absent imaging — nothing to reconcile

        out.append(Reconciliation(etiology=etio, prob=round(prob, 4), recon=recon,
                                  evidence=ev))
    return out


# ============================================================================ #
# Conflict detection (structural only — no amyloid; §8)
# ============================================================================ #

def detect_conflicts(diff: Differential, cards: list[FindingCard]) -> list[Conflict]:
    """Surface structural evidence conflicts. Order is stable; may be empty."""
    by_etio = _cards_by_etiology(cards)
    conflicts: list[Conflict] = []

    def metric_of(etio: str) -> Metric | None:
        c = by_etio.get(etio)
        return c.metric if c else None

    ad, nph = metric_of("AD"), metric_of("NPH")

    # 1 — strong AD signal without medial-temporal atrophy.
    if diff.p_ad >= P_AD_STRONG and _measured(ad) and not ad.abnormal:
        conflicts.append(Conflict(
            conflict_id="ad_signal_without_atrophy",
            name="AD signal without medial-temporal atrophy",
            description="A strong clinical AD signal is not matched by hippocampal "
                        "atrophy. Consider early/atypical AD, an AD mimic, or limits "
                        "of a single-timepoint volumetric.",
            severity=ConflictSeverity.IMPORTANT,
            evidence=[f"P(AD)={diff.p_ad:.2f} (≥{P_AD_STRONG}).",
                      f"Hippocampal Z {ad.value} (not {ad.comparator} {ad.threshold})."]))

    # 2 — medial-temporal atrophy without a clinical AD signal.
    if _measured(ad) and ad.abnormal and diff.p_ad < _PROB_GATE["AD"]:
        conflicts.append(Conflict(
            conflict_id="atrophy_without_ad_signal",
            name="Hippocampal atrophy without a clinical AD signal",
            description="Medial-temporal atrophy without a concordant clinical AD "
                        "signal suggests a non-AD process (e.g. LATE, hippocampal "
                        "sclerosis) or age-related change.",
            severity=ConflictSeverity.CAUTION,
            evidence=[f"Hippocampal Z {ad.value} ({ad.comparator} {ad.threshold}).",
                      f"P(AD)={diff.p_ad:.2f} (<{_PROB_GATE['AD']})."]))

    # 3 — ventriculomegaly without an NPH clinical signal.
    if _measured(nph) and nph.abnormal and diff.p("NPH") < _PROB_GATE["NPH"]:
        conflicts.append(Conflict(
            conflict_id="ventriculomegaly_without_nph",
            name="Ventriculomegaly without an NPH signal",
            description="An enlarged Evans-like index without an NPH clinical picture "
                        "may reflect atrophy ex vacuo rather than hydrocephalus; "
                        "correlate gait, continence and cognition.",
            severity=ConflictSeverity.CAUTION,
            evidence=[f"Evans-like index {nph.value} ({nph.comparator} {nph.threshold}).",
                      f"P(NPH)={diff.p('NPH'):.2f} (<{_PROB_GATE['NPH']})."]))

    # 4 — dementia-level stage with benign structural imaging.
    measured_metrics = [m for m in (ad, nph) if _measured(m)]
    if (diff.stage_top == "DE" and measured_metrics
            and not any(m.abnormal for m in measured_metrics)):
        conflicts.append(Conflict(
            conflict_id="severity_exceeds_imaging",
            name="Dementia stage exceeds structural imaging burden",
            description="A dementia-level clinical stage is not matched by the "
                        "structural findings assessed. Consider non-degenerative "
                        "contributors (mood, metabolic, medication), an unassessed "
                        "axis (e.g. vascular burden without FLAIR), and follow-up.",
            severity=ConflictSeverity.CAUTION,
            evidence=[f"Stage {diff.stage_top}; "
                      + ", ".join(f"{m.name} {m.value}" for m in measured_metrics)
                      + " all within normal limits."]))

    return conflicts


# ============================================================================ #
# Provenance + assembly
# ============================================================================ #

def _provenance(diff: Differential, cards: list[FindingCard]) -> list[str]:
    src = ("clinical features + MRI embedding (multimodal)" if diff.imaging_used
           else "clinical features only")
    prov = [f"Differential probabilities ← Xue 2024 (Nature Medicine) ADRD model, {src}."]
    for c in cards:
        if c.metric and c.metric.status is MetricStatus.MEASURED:
            prov.append(f"{c.metric.name} = {c.metric.value} ← {c.metric.reference}")
    return prov


def aggregate(
    case_id: str, differential: Differential, plan: ImagingPlan,
    cards: list[FindingCard],
) -> UnifiedEvidence:
    """Merge everything into the provenance-tagged S5 bundle (plain code, no LLM)."""
    return UnifiedEvidence(
        case_id=case_id,
        differential=differential,
        plan=plan,
        cards=cards,
        reconciliations=reconcile(differential, cards),
        conflicts=detect_conflicts(differential, cards),
        provenance=_provenance(differential, cards),
    )
