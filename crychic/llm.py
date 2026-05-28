"""Nemotron LLM client + the 4 pipeline prompts, with an offline fallback.

The three agentic touch-points (router rationale, report synthesis,
critic↔reviser loop) all route through here. Each call tries the
OpenAI-compatible Nemotron endpoint; if ``NEMOTRON_URL`` is unset or the request
fails, it degrades to a **deterministic template** so the whole pipeline — the
self-check loop included — still runs with no GPU or NIM.

The critic is special: its source of truth is :func:`cds_violations`, a pure
deterministic check of the 6 CDS principles. Online, the LLM critic's prose is
still backstopped by this checker, so "PASS" can never be hallucinated past a
report that is actually missing its sign-off footer.

Public surface (all coroutines):
    router_rationale(tier1, routing)            -> str
    write_report(ctx)                           -> str   (Reasoner)
    critique(markdown)                          -> list[str]  ([] == PASS)
    revise(markdown, violations, ctx)           -> str   (Reviser)
    build_report_context(...)                   -> ReportContext
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .schemas import (
    ClinicalPattern,
    Conflict,
    RoutingDecision,
    Tier1Result,
    Tier2Result,
)

# ============================================================================ #
# Fixed CDS boilerplate (principles #2 and #6)
# ============================================================================ #

_HEADER_MARK = "Clinical Decision Support — not a diagnosis"
_FOOTER_MARK = "Decision support only"
SIGN_OFF = "[✓ Agree & sign] [✏️ Edit] [✗ Disagree]"

CDS_HEADER = (
    f"> ⚕️ **{_HEADER_MARK}.** CRYCHIC organizes multi-modal evidence to assist a "
    "qualified clinician. It does not establish a diagnosis, prescribe, or direct "
    "care. The clinician makes and is responsible for the final decision."
)
CDS_FOOTER = (
    "---\n"
    f"> ⚕️ **{_FOOTER_MARK}.** This draft is not part of the medical record until a "
    "clinician reviews and signs it. All values are model-derived and must be "
    "verified against the source images and the full clinical context.\n\n"
    f"{SIGN_OFF}"
)

_REFERENCES = [
    "Xue et al., *Nature Medicine* 2024 — AI differential diagnosis of dementia.",
    "Klunk et al. 2015 — Centiloid standardization (GAAIN ≥20 positivity).",
    "NIA-AA 2018 — A/T/(N) biological framework.",
    "Wardlaw STRIVE — small-vessel-disease imaging standards.",
    "Boston v2.0 — CAA imaging diagnosis (requires SWI).",
]


# ============================================================================ #
# System prompts (used when a live Nemotron endpoint is configured)
# ============================================================================ #

ROUTER_SYSTEM = (
    "You are the routing rationale writer for CRYCHIC, a dementia clinical "
    "decision-support tool. Given Tier-1 screening probabilities and the imaging "
    "tools that hard-coded rules have already selected, write 2–4 sentences of "
    "clinical reasoning explaining WHY those tools are appropriate for this "
    "patient. Do not invent results. Do not diagnose or recommend treatment. "
    "Hedged, evidence-organizing language only."
)

REASONER_SYSTEM = (
    "You are the Reasoner for CRYCHIC. Compose a Markdown clinical-evidence "
    "summary from the structured evidence provided. You MUST obey all 6 CDS "
    "principles: (1) hedged language — never use 'diagnose' affirmatively or "
    "'recommend'; (2) keep the provided header and footer disclaimers verbatim; "
    "(3) every numeric value must carry its threshold and a reference citation; "
    "(4) include a non-empty differential/counter-evidence section and a "
    "limitations section; (5) present choices as '○' option bullets, never "
    "numbered recommendations; (6) keep the sign-off buttons in the footer. "
    "Organize evidence; do not decide."
)

CRITIC_SYSTEM = (
    "You are the Critic for CRYCHIC. Check the report against the 6 CDS "
    "principles. If it fully complies, output exactly 'PASS'. Otherwise output a "
    "bulleted list ('- ...') of specific violations, one per line, and nothing "
    "else."
)

REVISER_SYSTEM = (
    "You are the Reviser for CRYCHIC. Given a report and a list of CDS-principle "
    "violations, output a corrected full Markdown report that fixes every "
    "violation while preserving the evidence. Keep the header/footer disclaimers "
    "and sign-off buttons verbatim. Output only the report."
)


# ============================================================================ #
# Report context
# ============================================================================ #

@dataclass
class ReportContext:
    case_id: str
    tier1: Tier1Result
    routing: RoutingDecision
    tier2: Tier2Result
    pattern: ClinicalPattern
    conflicts: list[Conflict]


def build_report_context(
    case_id: str,
    tier1: Tier1Result,
    routing: RoutingDecision,
    tier2: Tier2Result,
    pattern: ClinicalPattern,
    conflicts: list[Conflict],
) -> ReportContext:
    return ReportContext(case_id, tier1, routing, tier2, pattern, conflicts)


# ============================================================================ #
# Nemotron HTTP transport (lazy, optional)
# ============================================================================ #

def _endpoint() -> str | None:
    return os.environ.get("NEMOTRON_URL") or None


def _model() -> str:
    return os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-nano-9b-v2")


def _api_key() -> str | None:
    """Bearer token for hosted Nemotron APIs (NVIDIA-hosted requires one).

    Self-hosted NIMs need no key, so this is optional; ``NEMOTRON_API_KEY`` wins,
    with ``NGC_API_KEY`` accepted as the conventional NVIDIA fallback.
    """
    return os.environ.get("NEMOTRON_API_KEY") or os.environ.get("NGC_API_KEY") or None


async def _chat(system: str, user: str, *, max_tokens: int = 1400) -> str:
    """One OpenAI-compatible chat completion. Raises if no endpoint / on error."""
    url = _endpoint()
    if not url:
        raise RuntimeError("NEMOTRON_URL not configured")
    import httpx

    headers = {}
    key = _api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": _model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    timeout = float(os.environ.get("NEMOTRON_TIMEOUT", "60"))
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ============================================================================ #
# Evidence serialization (for the LLM user message)
# ============================================================================ #

def _evidence_brief(ctx: ReportContext) -> str:
    t1, t2 = ctx.tier1, ctx.tier2
    lines = [
        f"CASE: {ctx.case_id}",
        f"TIER-1: stage_top={t1.stage_top}, etiology_top={t1.etiology_top}, "
        f"P(AD)={t1.p_ad:.2f}, P(MCI)={t1.p_mci:.2f}, P(VD)={t1.p_vd:.2f}, "
        f"P(FTD)={t1.p('FTD'):.2f}, P(PSY)={t1.p('PSY'):.2f}",
        f"ROUTING: tools={[t.value for t in ctx.routing.selected_tools]}; "
        f"rules={ctx.routing.fired_rules}",
    ]
    if t2.centiloid:
        c = t2.centiloid
        lines.append(f"CENTILOID: {c.centiloid} (≥{c.threshold} positive → "
                     f"{c.positive}); source={c.source.value}; ref={c.reference}")
    if t2.anatomy:
        a = t2.anatomy
        lines.append(f"ANATOMY: hippo_total={a.hippocampus_total_mm3}mm³ "
                     f"Z={a.hippocampus_zscore} (≤{a.atrophy_zscore_threshold} atrophy); "
                     f"vent_index={a.ventricular_index}; atrophy={a.dominant_atrophy}; "
                     f"n_labels={a.n_labels}; source={a.source.value}; ref={a.reference}")
    lines.append(f"PATTERN: #{ctx.pattern.pattern_id} {ctx.pattern.name} — "
                 f"{ctx.pattern.rationale} | evidence={ctx.pattern.supporting_evidence}")
    if ctx.conflicts:
        for cf in ctx.conflicts:
            lines.append(f"CONFLICT[{cf.severity.value}]: {cf.name} — {cf.description} "
                         f"| {cf.evidence}")
    else:
        lines.append("CONFLICT: none detected")
    lines += [
        "",
        "Header to keep verbatim:", CDS_HEADER,
        "Footer to keep verbatim:", CDS_FOOTER,
    ]
    return "\n".join(lines)


# ============================================================================ #
# Public API — each tries the LLM, falls back to a deterministic template
# ============================================================================ #

async def router_rationale(tier1: Tier1Result, routing: RoutingDecision) -> str:
    user = (
        f"Tier-1: P(AD)={tier1.p_ad:.2f}, P(MCI)={tier1.p_mci:.2f}, "
        f"P(VD)={tier1.p_vd:.2f}, stage={tier1.stage_top}.\n"
        f"Rules selected: {[t.value for t in routing.selected_tools]}.\n"
        f"Rule firings: {routing.fired_rules}.\n"
        "Explain why these imaging tools fit this patient."
    )
    try:
        return await _chat(ROUTER_SYSTEM, user, max_tokens=300)
    except Exception:
        return _template_router_rationale(tier1, routing)


async def write_report(ctx: ReportContext) -> str:
    try:
        out = await _chat(REASONER_SYSTEM, _evidence_brief(ctx))
        return _ensure_boilerplate(out)
    except Exception:
        return _template_report(ctx)


async def critique(markdown: str) -> list[str]:
    """Return CDS-principle violations; empty list == PASS.

    Deterministic checker is authoritative. When online, the LLM critic can add
    findings but cannot wave through a report the checker rejects.
    """
    violations = cds_violations(markdown)
    try:
        verdict = await _chat(CRITIC_SYSTEM, markdown, max_tokens=400)
        if verdict.strip().upper() != "PASS":
            for line in verdict.splitlines():
                line = line.strip().lstrip("-*•").strip()
                if line and line.upper() != "PASS" and line not in violations:
                    violations.append(line)
    except Exception:
        pass
    return violations


async def revise(markdown: str, violations: list[str], ctx: ReportContext) -> str:
    user = (
        "Violations to fix:\n" + "\n".join(f"- {v}" for v in violations)
        + "\n\nReport to fix:\n" + markdown
        + "\n\nEvidence (for reference):\n" + _evidence_brief(ctx)
    )
    try:
        out = await _chat(REVISER_SYSTEM, user)
        fixed = _ensure_boilerplate(out)
        # If the LLM's revision still fails, fall back to the compliant template.
        return fixed if not cds_violations(fixed) else _template_report(ctx)
    except Exception:
        return _template_report(ctx)


# ============================================================================ #
# Deterministic CDS-principle checker (the critic's source of truth)
# ============================================================================ #

# Directive / definitive phrasings that violate principle #1 (hedged language).
# Note we judge the BODY only, so the disclaimers' "does not establish a
# diagnosis" wording is never flagged.
_FORBIDDEN = [
    r"\brecommend(s|ed|ation|ations)?\b",
    r"\bthe diagnosis is\b",
    r"\bis diagnosed (with|as)\b",
    r"\bwe advise\b",
    r"\bprescrib(e|es|ed|ing)\b",
    r"\bstart (the )?patient on\b",
    r"\bmust (start|be started|prescribe)\b",
]


def _strip_boilerplate(markdown: str) -> str:
    """Body with the fixed header/footer disclaimers removed (for principle #1)."""
    body = markdown
    for chunk in (CDS_HEADER, CDS_FOOTER):
        body = body.replace(chunk, "")
    return body


def cds_violations(markdown: str) -> list[str]:
    """Check all 6 CDS principles. Empty list == compliant (PASS)."""
    v: list[str] = []
    body = _strip_boilerplate(markdown)
    low = markdown.lower()

    # 1 — hedged language.
    for pat in _FORBIDDEN:
        m = re.search(pat, body, flags=re.IGNORECASE)
        if m:
            v.append(f"Principle 1 (hedged language): forbidden phrasing {m.group(0)!r}.")
            break

    # 2 — header + footer disclaimers present.
    if _HEADER_MARK.lower() not in low:
        v.append("Principle 2: CDS header disclaimer missing.")
    if _FOOTER_MARK.lower() not in low:
        v.append("Principle 2: CDS footer disclaimer missing.")

    # 3 — traceable claims: references section + threshold symbols present.
    if "## references" not in low and "references" not in low:
        v.append("Principle 3: no References section for traceable claims.")
    if not re.search(r"[≥≤<>]", markdown):
        v.append("Principle 3: numeric claims lack thresholds (no ≥/≤/< present).")

    # 4 — counter-evidence: differential + limitations sections, non-empty.
    if not _section_nonempty(markdown, "differential"):
        v.append("Principle 4: differential / counter-evidence section empty or missing.")
    if not _section_nonempty(markdown, "limitation"):
        v.append("Principle 4: limitations section empty or missing.")

    # 5 — options not recommendations: '○' bullets, no numbered lists.
    if "○" not in markdown:
        v.append("Principle 5: options must use '○' bullets (none found).")
    if re.search(r"(?m)^\s*\d+\.\s", body):
        v.append("Principle 5: numbered list present — options must not be numbered.")

    # 6 — sign-off buttons in footer.
    if SIGN_OFF not in markdown:
        v.append("Principle 6: sign-off buttons missing from footer.")

    return v


def _section_nonempty(markdown: str, keyword: str) -> bool:
    """True if a heading containing ``keyword`` exists with body text under it."""
    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#") and keyword.lower() in line.lower():
            for nxt in lines[i + 1:]:
                if nxt.lstrip().startswith("#"):
                    break
                if nxt.strip():
                    return True
    return False


# ============================================================================ #
# Deterministic templates (offline fallback)
# ============================================================================ #

def _ensure_boilerplate(markdown: str) -> str:
    """Make sure header/footer survive an LLM round-trip (cheap guardrail)."""
    out = markdown
    if _HEADER_MARK not in out:
        out = CDS_HEADER + "\n\n" + out
    if SIGN_OFF not in out:
        out = out.rstrip() + "\n\n" + CDS_FOOTER
    return out


def _template_router_rationale(tier1: Tier1Result, routing: RoutingDecision) -> str:
    tools = ", ".join(t.value for t in routing.selected_tools) or "MONAI only"
    return (
        f"Tier-1 screening is most consistent with stage {tier1.stage_top} "
        f"(P(AD)={tier1.p_ad:.2f}, P(MCI)={tier1.p_mci:.2f}, P(VD)={tier1.p_vd:.2f}). "
        f"On that basis the rules selected: {tools}. Structural segmentation runs "
        "for every case to provide an anatomical baseline; amyloid and "
        "perivascular-space tools are added when the corresponding clinical "
        "signal crosses its threshold. These tools organize evidence for the "
        "clinician's review."
    )


def _fmt_t1(t1: Tier1Result) -> list[str]:
    return [
        f"- Most likely cognitive stage: **{t1.stage_top}** "
        f"(P(MCI)={t1.p_mci:.2f}, P(DE)={t1.p('DE'):.2f}, P(NC)={t1.p('NC'):.2f}).",
        f"- Leading etiology signal: **{t1.etiology_top}** "
        f"(P(AD)={t1.p_ad:.2f}, P(VD)={t1.p_vd:.2f}, P(FTD)={t1.p('FTD'):.2f}, "
        f"P(PSY)={t1.p('PSY'):.2f}).",
        "- These probabilities are non-exclusive and reflect possible comorbidity; "
        "they are consistent with, not confirmatory of, any single etiology "
        "(Xue 2024).",
    ]


def _fmt_t2(t2: Tier2Result) -> list[str]:
    out: list[str] = []
    if t2.centiloid:
        c = t2.centiloid
        tag = " *(synthetic placeholder)*" if c.source.value == "synthetic" else ""
        out.append(
            f"- **Amyloid PET (MYGO-Centiloid):** Centiloid **{c.centiloid}** "
            f"(positivity threshold ≥ {c.threshold}; {c.reference}) → "
            f"{'amyloid-positive' if c.positive else 'amyloid-negative'}{tag}.")
    if t2.anatomy:
        a = t2.anatomy
        z = "n/a" if a.hippocampus_zscore is None else a.hippocampus_zscore
        out.append(
            f"- **Structural (MONAI wholeBrainSeg):** hippocampal Z = {z} "
            f"(≤ {a.atrophy_zscore_threshold} = atrophy), ventricular index = "
            f"{a.ventricular_index}, dominant atrophy = {a.dominant_atrophy} "
            f"over {a.n_labels} segmented structures ({a.reference}).")
    return out or ["- No imaging tools returned evidence for this case."]


def _fmt_differential(ctx: ReportContext) -> list[str]:
    out: list[str] = []
    if ctx.conflicts:
        for cf in ctx.conflicts:
            out.append(f"- **{cf.name}** ({cf.severity.value}): {cf.description} "
                       f"Evidence: {'; '.join(cf.evidence)}")
    else:
        out.append("- No internal evidence conflicts were detected, which does not "
                   "exclude alternative etiologies below.")
    out.append("Alternatives the clinician may weigh:")
    out += [
        "○ A non-AD neurodegenerative process (e.g. FTD-spectrum, LATE) if the "
        "clinical course or atrophy emphasis diverges from the amyloid status.",
        "○ A vascular or mixed contribution, given the Tier-1 P(VD) signal.",
        "○ A reversible/functional contributor (mood, metabolic, medication) "
        "pending longitudinal follow-up.",
    ]
    return out


def _limitations(ctx: ReportContext) -> list[str]:
    lim = []
    if ctx.tier2.centiloid is None or ctx.tier2.centiloid.source.value == "synthetic":
        lim.append("- Amyloid burden is a synthetic placeholder / no PET was "
                   "quantified — amyloid status is not measured here.")
    lim += [
        "- No Tau PET input — tau trajectory subtypes cannot be distinguished.",
        "- No DaTscan / α-syn SAA — LBD comorbidity can be suggested, not confirmed.",
        "- No SWI or perivascular-space segmentation in v0.3 — CAA / vascular "
        "burden is only assessed via the Tier-1 P(VD) signal.",
        "- Single timepoint — pseudodementia cannot be fully excluded without "
        "longitudinal follow-up.",
        "- The Tier-1 model was trained on NACC; distribution shift applies to "
        "other cohorts (e.g. OASIS-3).",
    ]
    return lim


def _template_report(ctx: ReportContext) -> str:
    p = ctx.pattern
    parts: list[str] = [
        CDS_HEADER,
        "",
        f"# CRYCHIC Evidence Summary — `{ctx.case_id}`",
        "",
        "## Clinical screening (Tier-1)",
        *_fmt_t1(ctx.tier1),
        "",
        "## Imaging evidence (Tier-2)",
        *_fmt_t2(ctx.tier2),
        "",
        "## Most consistent pattern",
        f"- **Pattern {p.pattern_id} — {p.name}** (confidence: {p.confidence}).",
        f"- Rationale: {p.rationale}",
        *(f"- Supporting: {s}" for s in p.supporting_evidence),
        "",
        "## Differential considerations & counter-evidence",
        *_fmt_differential(ctx),
        "",
        "## Options for the clinician to consider",
        "○ Correlate this evidence with the history, exam, and prior imaging before "
        "any decision.",
        "○ Consider confirmatory work-up (e.g. CSF or additional PET) where the "
        "amyloid or vascular picture is uncertain.",
        "○ Consider longitudinal follow-up to clarify trajectory.",
        "",
        "## Limitations",
        *_limitations(ctx),
        "",
        "## References",
        *(f"- {r}" for r in _REFERENCES),
        "",
        CDS_FOOTER,
    ]
    return "\n".join(parts)
