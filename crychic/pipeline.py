"""Stage orchestration — the async core the MCP tools fire and poll.

``start_pipeline`` registers a case and launches :func:`run_pipeline` as a
background task, so ``run_crychic_pipeline`` can return a ``case_id`` instantly
while the work proceeds. The pipeline walks the documented stage sequence,
updating the shared :class:`~crychic.state.CaseState` at each step:

    screening → routing → imaging (parallel) → aggregating → reasoning
              → self_check_attempt_n ↔ revising_attempt_n → self_check_passed
              → complete | failed

Design choices that matter:

* **Imaging runs in parallel** via ``asyncio.gather``, and each tool is wrapped
  so a single tool's failure becomes a caveat on that slot, not a dead case —
  partial evidence still drives the aggregator and is surfaced as a limitation.
* **Routing is rule-driven** (CLAUDE.md §2.3); the LLM only writes the prose
  rationale. The execution path stays deterministic and auditable.
* **The self-check loop is genuinely variable-length** — it stops when the
  critic returns no violations, not after a fixed count (capped at
  ``MAX_SELF_CHECK_ITER`` to bound latency).
"""

from __future__ import annotations

import asyncio
import os
import traceback
from pathlib import Path

from . import aggregator, llm
from . import tier1_screening as t1
from . import tier2_imaging as t2
from .schemas import (
    CrychicReport,
    RoutingDecision,
    Tier1Result,
    Tier2Result,
    ToolName,
)
from .state import (
    STORE,
    CaseInputs,
    CaseState,
    Stage,
    revising_stage,
    self_check_stage,
)

# Keep references to background tasks so they are not garbage-collected.
_TASKS: set[asyncio.Task] = set()

# Routing thresholds (CLAUDE.md §2.3).
P_AD_CENTILOID = 0.30
P_MCI_CENTILOID = 0.40
P_VD_MUJICA = 0.20
P_AD_MUJICA = 0.50


def _max_iter() -> int:
    try:
        return max(1, int(os.environ.get("MAX_SELF_CHECK_ITER", "3")))
    except ValueError:
        return 3


# ============================================================================ #
# Entry point
# ============================================================================ #

def start_pipeline(inputs: CaseInputs) -> CaseState:
    """Register a case and launch the pipeline in the background."""
    state = STORE.create(inputs)
    task = asyncio.create_task(run_pipeline(state))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return state


async def run_pipeline(state: CaseState) -> None:
    """Run the full pipeline for one case, recording progress on ``state``."""
    try:
        state.set_stage(Stage.SCREENING)
        state.tier1 = await _run_tier1(state.inputs)

        state.set_stage(Stage.ROUTING)
        state.routing = _route(state.tier1, state.inputs)
        state.routing.rationale = await llm.router_rationale(state.tier1, state.routing)

        state.set_stage(Stage.IMAGING)
        state.tier2 = await _run_imaging(state)

        state.set_stage(Stage.AGGREGATING)
        state.pattern = aggregator.match_clinical_pattern(state.tier1, state.tier2)
        state.conflicts = aggregator.detect_conflicts(state.tier1, state.tier2)

        state.set_stage(Stage.REASONING)
        await _reason_and_self_check(state)

        state.complete()
    except Exception:  # never let a background task die silently
        state.fail(traceback.format_exc(limit=4))


# ============================================================================ #
# Stage 1 — screening
# ============================================================================ #

async def _run_tier1(inputs: CaseInputs) -> Tier1Result:
    """Tier-1 screen, degrading gracefully when the MRI path or free text fails.

    Preference order for the optional MRI: a precomputed embedding (fast), then
    the raw T1 (runs SwinUNETR if SSL weights are present). On any failure we
    retry clinical-only so screening is never blocked by imaging.
    """
    mri = inputs.mri_embedding_path or inputs.t1_path
    clinical = inputs.clinical
    # Free text that isn't a file path cannot feed the structured Xue model.
    if isinstance(clinical, str) and not Path(clinical).exists():
        clinical = {}

    for attempt_mri in (mri, None):
        try:
            return await t1.screen_async(clinical, mri=attempt_mri, explain=True)
        except Exception:
            continue
    # Last resort: clinical-only with whatever mapped (model returns its prior).
    return await t1.screen_async(clinical if isinstance(clinical, dict) else {},
                                 mri=None, explain=False)


# ============================================================================ #
# Stage 2 — routing (rules pick tools; LLM rationale added by caller)
# ============================================================================ #

def _route(tier1: Tier1Result, inputs: CaseInputs) -> RoutingDecision:
    selected: list[ToolName] = []
    fired: list[str] = []

    if tier1.p_ad >= P_AD_CENTILOID or tier1.p_mci >= P_MCI_CENTILOID:
        selected.append(ToolName.CENTILOID)
        fired.append(
            f"P(AD)={tier1.p_ad:.2f}≥{P_AD_CENTILOID} or "
            f"P(MCI)={tier1.p_mci:.2f}≥{P_MCI_CENTILOID} → centiloid"
        )
    if tier1.p_vd >= P_VD_MUJICA or tier1.p_ad >= P_AD_MUJICA:
        selected.append(ToolName.MUJICA)
        fired.append(
            f"P(VD)={tier1.p_vd:.2f}≥{P_VD_MUJICA} or "
            f"P(AD)={tier1.p_ad:.2f}≥{P_AD_MUJICA} → mujica"
        )
    # MONAI always — when a structural T1 is available to segment.
    if inputs.t1_path:
        selected.append(ToolName.MONAI)
        fired.append("always (T1 present) → monai")
    else:
        fired.append("MONAI skipped — no T1 volume supplied")

    return RoutingDecision(selected_tools=selected, fired_rules=fired)


# ============================================================================ #
# Stage 3 — parallel imaging
# ============================================================================ #

async def _run_imaging(state: CaseState) -> Tier2Result:
    t1r = state.tier1
    inp = state.inputs
    selected = set(state.routing.selected_tools)
    seed = t1r.input_id or state.case_id

    amyloid_prior = max(t1r.p_ad, t1r.p_mci)
    vascular_prior = t1r.p_vd
    caa_prior = max(t1r.p_vd, 0.5 * t1r.p_ad)

    async def _centiloid():
        if ToolName.CENTILOID not in selected:
            return None
        r = await t2.run_centiloid_async(
            inp.pet_path, inp.tracer, amyloid_prior=amyloid_prior, seed_key=seed)
        state.mark_tool_done("centiloid")
        return r

    async def _mujica():
        if ToolName.MUJICA not in selected:
            return None
        r = await t2.run_epvs_async(
            inp.t1_path, vascular_prior=vascular_prior, caa_prior=caa_prior,
            seed_key=seed)
        state.mark_tool_done("mujica")
        return r

    async def _monai():
        if ToolName.MONAI not in selected or not inp.t1_path:
            return None
        r = await t2.run_anatomy_async(inp.t1_path)
        state.mark_tool_done("monai")
        return r

    centiloid, epvs, anatomy = await asyncio.gather(
        _guard(_centiloid()), _guard(_mujica()), _guard(_monai())
    )
    return Tier2Result(centiloid=centiloid, epvs=epvs, anatomy=anatomy)


async def _guard(coro):
    """Run an imaging coroutine; swallow its failure (logged, not fatal)."""
    try:
        return await coro
    except Exception:
        # A tool's absence/failure becomes missing evidence + a surfaced
        # limitation downstream, not a dead case.
        return None


# ============================================================================ #
# Stage 5 — reasoning + self-check loop
# ============================================================================ #

async def _reason_and_self_check(state: CaseState) -> None:
    ctx = llm.build_report_context(
        state.case_id, state.tier1, state.routing, state.tier2,
        state.pattern, state.conflicts,
    )
    markdown = await llm.write_report(ctx)
    max_iter = _max_iter()

    for attempt in range(1, max_iter + 1):
        state.set_stage(self_check_stage(attempt))
        violations = await llm.critique(markdown)

        if not violations:
            state.report = CrychicReport(
                markdown=markdown, self_check_passed=True, iterations=attempt)
            state.set_stage(Stage.SELF_CHECK_PASSED)
            return

        if attempt == max_iter:
            state.report = CrychicReport(
                markdown=markdown, self_check_passed=False, iterations=attempt,
                remaining_violations=violations,
                warning=f"Self-check did not converge in {max_iter} iterations; "
                        "returned with unresolved CDS findings.",
            )
            return

        state.set_stage(revising_stage(attempt))
        markdown = await llm.revise(markdown, violations, ctx)
