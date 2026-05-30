"""S4d translate + S6 reason — turn metrics into finding cards and a CDS report.

Two responsibilities, both agentic but tightly guarded:

* **S4d translate** — a :class:`~crychic.schemas.Metric` becomes a
  :class:`~crychic.schemas.FindingCard`. The impression ``sentence`` is built
  deterministically with the metric's digits injected as **fixed tokens** (Inv #2):
  the language model may polish the qualitative wording, but a guard rejects any
  rephrase that introduces a numeric token the metric did not contain — so the
  model is structurally unable to put a wrong number in a finding. A normal result
  still yields a card (Inv #7); an unmeasured/abstained axis yields an abstain
  card (Inv #8).
* **S6 reason** — compose the clinician-facing report from the unified evidence,
  obeying the CDS output rules (Inv #9). Online it drafts with the LLM (backstopped
  by :mod:`crychic.cds_guard`); offline it uses a deterministic template. The
  critic↔reviser self-check loop is driven by the pipeline; its source of truth is
  the deterministic ``cds_violations`` checker.
"""

from __future__ import annotations

import re

from .. import cds_guard, checks, llm_client, overlay
from ..schemas import (
    Conflict,
    FindingCard,
    ImagingCheck,
    KeySlice,
    Metric,
    MetricStatus,
    Modality,
    ReconClass,
    Reconciliation,
    UnifiedEvidence,
)

ETIOLOGY_NAME = {
    "AD": "Alzheimer's disease",
    "LBD": "Lewy body / Parkinson's dementia",
    "VD": "Vascular dementia",
    "PRD": "Prion disease",
    "FTD": "Frontotemporal dementia",
    "NPH": "Normal-pressure hydrocephalus",
    "SEF": "Systemic / environmental factors",
    "PSY": "Psychiatric contribution",
    "TBI": "Traumatic brain injury",
    "ODE": "Other / structural etiologies",
}


# ============================================================================ #
# S4d — translate a Metric into a guardrailed FindingCard
# ============================================================================ #

def _numeric_tokens(text: str) -> set[str]:
    """All maximal digit-runs in ``text`` (e.g. {'1.5', '0.30', '2'})."""
    return set(re.findall(r"\d+\.?\d*", text))


def sentence_within_number_budget(candidate: str, allowed_source: str) -> bool:
    """True iff ``candidate`` introduces no digit not present in ``allowed_source``.

    The guard behind Inv #2: a polished sentence may only reuse the numbers that
    the deterministic, metric-derived sentence already contained.
    """
    return _numeric_tokens(candidate).issubset(_numeric_tokens(allowed_source))


def _measured_sentence(m: Metric) -> str:
    """Deterministic impression with the metric's digits as the only numbers.

    The per-etiology phrasing lives on the :class:`~crychic.checks.CheckSpec`
    sentence templates (single source of truth); here we only inject the metric's
    own tokens (Inv #2).
    """
    spec = checks.CHECKS_BY_ETIOLOGY.get(m.etiology)
    unit = f" {m.unit}" if m.unit else ""
    if spec is None:  # generic fallback for an etiology with no registered check
        return f"{m.name}: {m.value}{unit} (threshold {m.comparator} {m.threshold})."
    return spec.sentence(m.value, m.threshold, m.comparator, unit, abnormal=m.abnormal)


def translate(
    metric: Metric, *, check: ImagingCheck | None = None,
    t1_path: str | None = None, flair_path: str | None = None,
    out_dir: str | None = None,
) -> FindingCard:
    """S4d: a Metric → a radiology-style FindingCard (value+threshold+ref, Inv #7).

    The ``sentence`` is built deterministically from the metric, so its only
    numbers are the metric's own (Inv #2). A future LLM "qualitative polish" is
    permitted only if it passes :func:`sentence_within_number_budget` — the guard
    that forbids introducing any digit the metric did not contain.
    """
    etio = metric.etiology
    title = f"{ETIOLOGY_NAME.get(etio, etio)} — {metric.name}"

    if metric.status is not MetricStatus.MEASURED:
        why = metric.caveats[0] if metric.caveats else "not assessed."
        return FindingCard(
            etiology=etio, title=title, metric=metric, polarity="abstain",
            sentence=(f"{ETIOLOGY_NAME.get(etio, etio)}: not assessed from imaging — "
                      f"{why} Imaging neither confirms nor excludes it."),
            references=[metric.reference] if metric.reference else [],
        )

    sentence = _measured_sentence(metric)

    key_slice, png = None, None
    rendered = None
    if check is ImagingCheck.FAZEKAS and flair_path:
        # The VD finding's key slice is a FLAIR overlay (WMH highlighted), not a T1 one.
        rendered = overlay.render_wmh_overlay(flair_path, out_dir=out_dir)
    elif check is not None and t1_path:
        # render_overlay returns None when the check has no T1 key-slice (Inv #7).
        rendered = overlay.render_overlay(t1_path, check, out_dir=out_dir)
    if rendered:
        key_slice = KeySlice(plane=rendered["plane"], index=rendered["index"])
        png = rendered["png_path"]

    return FindingCard(
        etiology=etio, title=title, metric=metric,
        polarity="supporting" if metric.abnormal else "negative",
        sentence=sentence, key_slice=key_slice, overlay_png_path=png,
        references=[metric.reference] if metric.reference else [],
    )


def abstain_card(etiology: str, modalities: set[Modality]) -> FindingCard:
    """A finding card for an etiology with no imaging assessment (Inv #8)."""
    name = ETIOLOGY_NAME.get(etiology, etiology)
    if etiology == "VD" and Modality.FLAIR not in modalities:
        reason = ("FLAIR was not provided, so white-matter-hyperintensity burden "
                  "was not assessed.")
    elif etiology == "FTD":
        reason = ("frontotemporal volumetry was retired in this build (its normative "
                  "reference was synthetic, not atlas-matched), so FTD is left to the "
                  "clinical features.")
    elif etiology == "ODE":
        reason = "the tumour / structural-lesion (BraTS) axis is not enabled in this build."
    else:
        reason = "no off-the-shelf structural imaging correlate exists for this etiology."
    return FindingCard(
        etiology=etiology, title=f"{name} — no imaging correlate", metric=None,
        polarity="abstain",
        sentence=(f"{name}: not assessed from imaging — {reason} Imaging neither "
                  "confirms nor excludes it; rely on the clinical features."),
    )


# ============================================================================ #
# S6 — the report (LLM with deterministic fallback) + self-check helpers
# ============================================================================ #

_REASONER_SYSTEM = (
    "You are the Reasoner for CRYCHIC. Compose a Markdown clinical-evidence summary "
    "from the structured evidence provided. Obey all CDS rules: (1) hedged language "
    "— never 'diagnose' affirmatively or 'recommend'; (2) keep the provided header "
    "and footer verbatim; (3) every numeric value keeps its threshold and a "
    "reference; (4) include non-empty Differential and Limitations sections; "
    "(5) present choices as '○' option bullets, never numbered; (6) keep the footer's "
    "statement that the draft requires clinician sign-off. Do NOT add any "
    "'[Agree & sign] / [Edit] / [Disagree]' line — sign-off is handled by the "
    "application UI. Use ONLY the numbers given — invent no values. Organize "
    "evidence; do not decide."
)
_CRITIC_SYSTEM = (
    "You are the Critic for CRYCHIC. Check the report against the 6 CDS principles. "
    "If it fully complies, output exactly 'PASS'. Otherwise output a bulleted list "
    "('- ...') of specific violations, one per line, and nothing else."
)
_REVISER_SYSTEM = (
    "You are the Reviser for CRYCHIC. Given a report and CDS violations, output a "
    "corrected full Markdown report that fixes every violation while preserving the "
    "evidence and every number. Keep the header and footer verbatim; do not add a "
    "'[Agree & sign]' sign-off line (the application UI handles sign-off)."
)


async def write_report(unified: UnifiedEvidence) -> str:
    """S6 draft: LLM from the evidence brief, else the deterministic template."""
    try:
        out = await llm_client.chat(_REASONER_SYSTEM, _evidence_brief(unified))
        return cds_guard.ensure_boilerplate(out)
    except Exception:
        return template_report(unified)


async def critique(markdown: str) -> list[str]:
    """CDS violations; [] == PASS. Deterministic checker is authoritative."""
    violations = cds_guard.cds_violations(markdown)
    try:
        verdict = await llm_client.chat(_CRITIC_SYSTEM, markdown, max_tokens=400)
        if verdict.strip().upper() != "PASS":
            for line in verdict.splitlines():
                line = line.strip().lstrip("-*•").strip()
                if line and line.upper() != "PASS" and line not in violations:
                    violations.append(line)
    except Exception:
        pass
    return violations


async def revise(markdown: str, violations: list[str], unified: UnifiedEvidence) -> str:
    user = ("Violations to fix:\n" + "\n".join(f"- {v}" for v in violations)
            + "\n\nReport to fix:\n" + markdown
            + "\n\nEvidence (for reference):\n" + _evidence_brief(unified))
    try:
        fixed = cds_guard.ensure_boilerplate(await llm_client.chat(_REVISER_SYSTEM, user))
        return fixed if not cds_guard.cds_violations(fixed) else template_report(unified)
    except Exception:
        return template_report(unified)


# --- evidence serialization for the LLM --------------------------------------

def _evidence_brief(u: UnifiedEvidence) -> str:
    d = u.differential
    screen_tag = "clinical + MRI embedding" if d.imaging_used else "clinical only"
    lines = [
        f"CASE: {u.case_id}",
        f"SCREEN ({screen_tag}): stage={d.stage_top}, etiology_top={d.etiology_top}, "
        f"P(AD)={d.p_ad:.2f}, P(MCI)={d.p_mci:.2f}, P(VD)={d.p_vd:.2f}, "
        f"P(FTD)={d.p('FTD'):.2f}, P(NPH)={d.p('NPH'):.2f}",
        f"PLAN: checks={[c.value for c in u.plan.checks]}; rules={u.plan.fired_rules}; "
        f"abstained={u.plan.abstained}",
        "FINDING CARDS (use these sentences/numbers verbatim — do not alter digits):",
    ]
    for c in u.cards:
        lines.append(f"  - [{c.polarity}] {c.title}: {c.sentence}")
    lines.append("RECONCILIATION:")
    for r in u.reconciliations:
        lines.append(f"  - {r.etiology} ({r.prob:.2f}) → {r.recon.value}: {'; '.join(r.evidence)}")
    if u.conflicts:
        for cf in u.conflicts:
            lines.append(f"CONFLICT[{cf.severity.value}]: {cf.name} — {cf.description}")
    else:
        lines.append("CONFLICT: none detected")
    lines += ["", "Header to keep verbatim:", cds_guard.CDS_HEADER,
              "Footer to keep verbatim:", cds_guard.CDS_FOOTER]
    return "\n".join(lines)


# --- deterministic template (offline fallback, always CDS-compliant) ----------

_RECON_BLURB = {
    ReconClass.CONCORDANT: "clinical signal supported by imaging",
    ReconClass.DISCORDANT: "clinical signal NOT supported by imaging — flag",
    ReconClass.CLINICAL_ONLY: "clinical signal with no imaging axis available",
    ReconClass.INCIDENTAL: "imaging finding without a strong clinical signal",
}


def template_report(u: UnifiedEvidence) -> str:
    d = u.differential
    card_lines = [_fmt_card(c) for c in u.cards] or [
        "- No imaging tools returned evidence for this case."]
    recon_lines = [
        f"- **{r.etiology}** ({r.prob:.2f}) — _{r.recon.value}_: "
        f"{_RECON_BLURB[r.recon]}. {' '.join(r.evidence)}"
        for r in u.reconciliations] or ["- No etiology crossed the reconciliation thresholds."]
    if d.imaging_used:
        screen_heading = "## Dementia screening (Xue 2024 — clinical + MRI, multimodal)"
        independence_note = (
            "- Probabilities are non-exclusive and consistent with, not confirmatory "
            "of, any single etiology; the MRI embedding informs them, so the "
            "structural findings below are a consistency cross-check on "
            "imaging-informed probabilities, not a fully independent axis.")
    else:
        screen_heading = "## Clinical screening (Xue 2024 — clinical features only)"
        independence_note = (
            "- Probabilities are non-exclusive and consistent with, not confirmatory "
            "of, any single etiology; imaging is assessed independently (no MRI is "
            "fed to the clinical model).")
    parts: list[str] = [
        cds_guard.CDS_HEADER, "",
        f"# CRYCHIC Evidence Summary — `{u.case_id}`", "",
        screen_heading,
        f"- Most likely cognitive stage: **{d.stage_top}** "
        f"(P(MCI)={d.p_mci:.2f}, P(DE)={d.p('DE'):.2f}, P(NC)={d.p('NC'):.2f}).",
        f"- Leading etiology signal: **{d.etiology_top}** "
        f"(P(AD)={d.p_ad:.2f}, P(VD)={d.p_vd:.2f}, P(FTD)={d.p('FTD'):.2f}, "
        f"P(NPH)={d.p('NPH'):.2f}).",
        independence_note,
        "",
        "## Imaging findings",
        *card_lines,
        "",
        "## Reconciliation (clinical signal vs imaging)",
        *recon_lines,
        "",
        "## Differential considerations & counter-evidence",
        *_fmt_differential(u),
        "",
        "## Options for the clinician to consider",
        "○ Correlate this evidence with the history, examination, and prior imaging "
        "before any decision.",
        "○ Consider confirmatory work-up where an axis is uncertain or unavailable "
        "(e.g. FLAIR for vascular burden, CSF/PET where indicated).",
        "○ Consider longitudinal follow-up to clarify trajectory.",
        "",
        "## Limitations",
        *_limitations(u),
        "",
        "## References",
        *(f"- {r}" for r in cds_guard.REFERENCES),
        "",
        cds_guard.CDS_FOOTER,
    ]
    return "\n".join(parts)


def _fmt_card(c: FindingCard) -> str:
    tag = {"supporting": "🟠", "negative": "🟢", "abstain": "⚪"}.get(c.polarity, "•")
    refs = f" ({'; '.join(c.references)})" if c.references else ""
    slice_note = (f" Key slice: {c.key_slice.plane} #{c.key_slice.index}."
                  if c.key_slice else "")
    return f"- {tag} **{c.title}** — {c.sentence}{slice_note}{refs}"


def _fmt_differential(u: UnifiedEvidence) -> list[str]:
    out: list[str] = []
    if u.conflicts:
        for cf in u.conflicts:
            out.append(f"- **{cf.name}** ({cf.severity.value}): {cf.description} "
                       f"Evidence: {'; '.join(cf.evidence)}")
    else:
        out.append("- No internal evidence conflicts were detected, which does not "
                   "exclude the alternatives below.")
    out.append("Alternatives the clinician may weigh:")
    out += [
        "○ A non-AD neurodegenerative process (e.g. FTD-spectrum, LATE) if the course "
        "or atrophy emphasis diverges from the hippocampal picture.",
        "○ A vascular or mixed contribution, particularly if the clinical vascular "
        "signal is elevated (FLAIR would clarify).",
        "○ A reversible / functional contributor (mood, metabolic, medication) "
        "pending longitudinal follow-up.",
    ]
    return out


def _limitations(u: UnifiedEvidence) -> list[str]:
    if u.differential.imaging_used:
        axis_note = ("- The Xue 2024 differential is multimodal (clinical + MRI "
                     "embedding); the structural finding cards therefore consistency-"
                     "check imaging-informed probabilities rather than supply a fully "
                     "independent imaging axis. Their numbers are still computed "
                     "independently in geometry code.")
    else:
        axis_note = ("- The clinical differential (Xue 2024) uses clinical features "
                     "only; imaging is an independent axis, so concordance is a "
                     "genuine cross-check.")
    lim = [
        axis_note,
        "- Structural metrics use an internal normative reference, not an "
        "age/sex/ICV-matched atlas; treat magnitudes as approximate.",
        "- The Evans-like index is a coarse screening proxy for the radiological "
        "Evans index (no skull in a brain-only segmentation) and is biased low.",
    ]
    if any(c.etiology == "VD" and c.polarity == "abstain" for c in u.cards):
        lim.append("- No FLAIR supplied — vascular (WMH/Fazekas) burden was not "
                   "assessed from imaging.")
    lim += [
        "- Several etiologies (LBD, prion, psychiatric, systemic, TBI) and tumour "
        "have no structural correlate here and are left to the clinical features.",
        "- Single timepoint — a functional/affective contributor cannot be excluded "
        "without longitudinal follow-up.",
        "- The clinical model was trained on NACC; distribution shift applies to "
        "other cohorts (e.g. OASIS-3).",
    ]
    return lim
