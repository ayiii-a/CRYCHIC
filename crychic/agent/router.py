"""S3 — the router: the one real LLM decision in the pipeline (CLAUDE.md Inv #5).

Given the clinical differential and which imaging modalities are present, decide
*which* structural checks to dispatch. This is the single place the LLM chooses;
everything downstream executes in fixed order.

How the decision is kept both agentic and auditable:

* **The free T1 structural baseline always runs** when a T1 is present — all of
  hippocampal Z (AD), the Evans-like index (NPH), and frontotemporal Z (FTD) come
  from the *one* segmentation at ~zero marginal cost (Inv #6), and a normal result
  is itself a useful negative finding (Inv #7). So these are a floor, not a gate.
* **The LLM's genuine decision is over the extra-cost, modality-gated axis** —
  whether to dispatch the WMH/Fazekas (VD) check, which needs FLAIR and a separate
  model. It chooses from the menu of available checks; it can never select a FLAIR
  check when there is no FLAIR.
* **Deterministic rules** provide the audit trail (``fired_rules``) — which clinical
  signal emphasizes which axis — and the offline fallback used verbatim when no
  endpoint is configured or the call fails.
* **Abstentions are explicit** (Inv #8): PRD/SEF/PSY/TBI/LBD have no structural
  correlate and ODE's tumor axis is not built, so they are recorded as abstained;
  VD is abstained when no FLAIR (or the router declines the WMH model).

The router never emits a number — it emits a plan.
"""

from __future__ import annotations

import json
import re

from .. import llm_client
from ..schemas import (
    ABSTAIN_ETIOLOGIES,
    CHECK_ETIOLOGY,
    CHECK_MODALITY,
    Differential,
    ImagingCheck,
    ImagingPlan,
    Modality,
)

# Routing thresholds (named so the report can cite them).
P_AD_GATE = 0.30
P_MCI_GATE = 0.40
P_NPH_GATE = 0.15
P_VD_GATE = 0.20

_NICHE_ABSTAIN = ("ODE",)  # tumor/BraTS axis — niche, not built (§4)

_ROUTER_SYSTEM = (
    "You are the imaging router for CRYCHIC, a dementia decision-support tool. "
    "Given clinical screening probabilities and a menu of available structural "
    "imaging checks, decide which checks are worth running for THIS patient. "
    "Return ONLY a JSON array of check ids (a subset of the menu). Prefer checks "
    "whose etiology the clinical signal supports; the free T1 structural baseline "
    "is cheap and worth keeping. Do not invent checks, do not add prose."
)


def available_modalities(t1_path: str | None, flair_path: str | None) -> set[Modality]:
    mods: set[Modality] = set()
    if t1_path:
        mods.add(Modality.T1)
    if flair_path:
        mods.add(Modality.FLAIR)
    return mods


def _t1_baseline(available: list[ImagingCheck]) -> list[ImagingCheck]:
    """The free T1 checks (run whenever a T1 is present — Inv #6/#7)."""
    return [c for c in available if CHECK_MODALITY[c] is Modality.T1]


def _rule_checks(diff: Differential, available: list[ImagingCheck]) -> tuple[list[ImagingCheck], list[str]]:
    """Deterministic plan + human-readable firings (audit trail / offline fallback).

    The T1 baseline always runs; FAZEKAS (FLAIR) is the gated extra. ``fired_rules``
    records which clinical signal emphasizes which axis.
    """
    fired: list[str] = []
    picked: list[ImagingCheck] = list(_t1_baseline(available))
    if picked:
        fired.append("T1 present → free structural baseline: "
                     + ", ".join(c.value for c in picked))

    # clinical-signal emphasis notes (annotation, not a gate, for T1 checks)
    if ImagingCheck.HIPPO_Z in picked and (diff.p_ad >= P_AD_GATE or diff.impaired):
        fired.append(f"P(AD)={diff.p_ad:.2f}≥{P_AD_GATE} / stage={diff.stage_top} "
                     "→ AD (hippo_z) emphasized")
    if ImagingCheck.EVANS in picked and diff.p("NPH") >= P_NPH_GATE:
        fired.append(f"P(NPH)={diff.p('NPH'):.2f}≥{P_NPH_GATE} → NPH (evans) emphasized")

    # FAZEKAS — the gated, FLAIR-only extra (the router's genuine choice).
    if ImagingCheck.FAZEKAS in available:
        if diff.p_vd >= P_VD_GATE:
            picked.append(ImagingCheck.FAZEKAS)
            fired.append(f"P(VD)={diff.p_vd:.2f}≥{P_VD_GATE} + FLAIR → fazekas (VD)")
        else:
            fired.append(f"FLAIR present but P(VD)={diff.p_vd:.2f}<{P_VD_GATE} "
                         "→ fazekas optional")
    elif diff.p_vd >= P_VD_GATE:
        fired.append(f"P(VD)={diff.p_vd:.2f}≥{P_VD_GATE} but no FLAIR → VD not assessed")

    return picked, fired


def _parse_llm_checks(text: str, available: list[ImagingCheck]) -> list[ImagingCheck] | None:
    """Parse a JSON array of check ids; keep only available ones. None on failure."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        ids = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    avail = {c.value: c for c in available}
    chosen = [avail[i] for i in ids if isinstance(i, str) and i in avail]
    return chosen


def _compute_abstained(checks: list[ImagingCheck]) -> list[str]:
    """Etiologies CRYCHIC will not speak to from imaging (Inv #8)."""
    covered = {CHECK_ETIOLOGY[c] for c in checks}
    abstained = set(ABSTAIN_ETIOLOGIES) | set(_NICHE_ABSTAIN)
    for check, etio in CHECK_ETIOLOGY.items():  # structural etiologies not measured
        if etio not in covered:
            abstained.add(etio)
    return sorted(abstained)


async def route(
    diff: Differential, t1_path: str | None, flair_path: str | None,
) -> ImagingPlan:
    """Decide the imaging plan (S3). LLM chooses; rules audit + backstop + floor."""
    available = [c for c in ImagingCheck if CHECK_MODALITY[c] in
                 available_modalities(t1_path, flair_path)]
    rule_checks, fired = _rule_checks(diff, available)

    checks = rule_checks
    rationale = _template_rationale(diff, checks)

    if available and llm_client.online():
        menu = ", ".join(f"{c.value} ({CHECK_ETIOLOGY[c]})" for c in available)
        user = (
            f"P(AD)={diff.p_ad:.2f} P(MCI)={diff.p_mci:.2f} P(VD)={diff.p_vd:.2f} "
            f"P(FTD)={diff.p('FTD'):.2f} P(NPH)={diff.p('NPH'):.2f}; "
            f"stage={diff.stage_top}.\nAvailable checks: [{menu}].\n"
            "Return the JSON array of check ids to run."
        )
        try:
            llm_choice = _parse_llm_checks(
                await llm_client.chat(_ROUTER_SYSTEM, user, max_tokens=120), available)
            if llm_choice is not None:
                # The LLM decides the gated extras; the free T1 baseline always runs
                # (Inv #6/#7), so it is unioned in and can never be dropped.
                checks = list(set(llm_choice) | set(_t1_baseline(available)))
                rationale = await _llm_rationale(diff, checks)
        except Exception:
            pass  # offline / failure → keep the deterministic rule plan

    # stable order by enum definition
    checks = [c for c in ImagingCheck if c in set(checks)]
    return ImagingPlan(checks=checks, rationale=rationale, fired_rules=fired,
                       abstained=_compute_abstained(checks))


# --- prose rationale (LLM, with a deterministic fallback) ---------------------

_RATIONALE_SYSTEM = (
    "You write a 2–3 sentence clinical rationale for which structural imaging "
    "checks CRYCHIC will run, given the screening probabilities. Hedged, "
    "evidence-organizing language; do not diagnose, do not recommend treatment."
)


async def _llm_rationale(diff: Differential, checks: list[ImagingCheck]) -> str:
    ids = ", ".join(c.value for c in checks) or "none"
    try:
        return await llm_client.chat(
            _RATIONALE_SYSTEM,
            f"Probabilities: P(AD)={diff.p_ad:.2f}, P(MCI)={diff.p_mci:.2f}, "
            f"P(VD)={diff.p_vd:.2f}, P(FTD)={diff.p('FTD'):.2f}, "
            f"P(NPH)={diff.p('NPH'):.2f}, stage={diff.stage_top}. "
            f"Checks selected: {ids}. Explain why.",
            max_tokens=220)
    except Exception:
        return _template_rationale(diff, checks)


def _template_rationale(diff: Differential, checks: list[ImagingCheck]) -> str:
    ids = ", ".join(c.value for c in checks) or "no imaging checks"
    return (
        f"Clinical screening is most consistent with stage {diff.stage_top} "
        f"(P(AD)={diff.p_ad:.2f}, P(VD)={diff.p_vd:.2f}, P(FTD)={diff.p('FTD'):.2f}, "
        f"P(NPH)={diff.p('NPH'):.2f}). On that basis the following structural "
        f"checks are dispatched: {ids}. The T1 segmentation is free, so the "
        "structural baseline runs whenever a T1 is available; non-structural "
        "etiologies are left to the clinical features. These checks organize "
        "imaging evidence for the clinician's review."
    )
