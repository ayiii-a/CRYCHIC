"""The in-process spine — S1→S7 in fixed order (CLAUDE.md §3).

This is the runnable orchestrator the web UI and scripts use; it calls the same
compute the MCP servers wrap (xue / segmentation / wmh) and the same agent logic
the NAT workflow would (extract / router / reasoner), as plain Python — so a demo
runs without spawning servers or a NAT install.

The flow is a fixed spine with exactly one LLM decision (the router, S3); it is
NOT a free-roaming ReAct loop (Inv #5):

    extracting → screening → routing → imaging → translating → aggregating
               → reasoning → self_check_attempt_n ↔ revising_attempt_n
               → self_check_passed → complete | failed

Design choices that matter:

* **Blocking inference runs in worker threads** (``asyncio.to_thread``) so polling
  stays responsive; results are assigned back on the loop thread.
* **A tool failure becomes a caveat / an UNAVAILABLE metric, never a dead case** —
  the missing axis surfaces as an honest abstain card, not a fabricated number.
* **The self-check loop is genuinely variable-length** — it stops when the critic
  returns no violations (capped by ``MAX_SELF_CHECK_ITER`` to bound latency).
"""

from __future__ import annotations

import asyncio
import os
import traceback

from . import aggregate, segmentation, wmh, xue
from .agent import extract, reasoner, router
from .schemas import (
    CHECK_ETIOLOGY,
    CHECK_MODALITY,
    CrychicReport,
    FindingCard,
    ImagingCheck,
    Metric,
    MetricStatus,
    Modality,
)
from .state import (
    STORE,
    CaseInputs,
    CaseState,
    Stage,
    revising_stage,
    self_check_stage,
)

_TASKS: set[asyncio.Task] = set()  # keep background tasks from being GC'd

_T1_CHECKS = {c for c, m in CHECK_MODALITY.items() if m is Modality.T1}


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
    """Run the full S1→S7 spine for one case, recording progress on ``state``."""
    try:
        inp = state.inputs

        # S1 — extract clinical features (LLM for prose; pass-through for a dict).
        state.set_stage(Stage.EXTRACTING)
        state.features = await extract.extract_features_async(inp.clinical)

        # S2 — the differential (multimodal: clinical + MRI embedding when present).
        state.set_stage(Stage.SCREENING)
        state.differential = await xue.screen_async(
            state.features.features, mri=inp.mri_emb_path, explain=True)
        state.differential.caveats += state.features.caveats
        state.mark_tool_done("xue_predict")

        # S3 — the one LLM decision: which imaging checks to dispatch.
        state.set_stage(Stage.ROUTING)
        state.plan = await router.route(state.differential, inp.t1_path, inp.flair_path)

        # S4a/b/c — imaging compute (segment once; derive each metric).
        state.set_stage(Stage.IMAGING)
        age, sex = _age_sex(state.features.features)
        metrics, tools, caveats = await asyncio.to_thread(
            _run_imaging, state.plan, inp, age, sex)
        for t in tools:
            state.mark_tool_done(t)
        state.differential.caveats += caveats

        # S4d — translate metrics into guardrailed finding cards (+ abstain cards).
        state.set_stage(Stage.TRANSLATING)
        state.cards = await asyncio.to_thread(_translate_all, metrics, state.plan, inp)

        # S5 — aggregate (plain code): merge, reconcile, conflicts, provenance.
        state.set_stage(Stage.AGGREGATING)
        state.unified = aggregate.aggregate(
            state.case_id, state.differential, state.plan, state.cards)

        # S6 — reason + self-check loop.
        state.set_stage(Stage.REASONING)
        await _reason_and_self_check(state)

        state.complete()
    except Exception:  # never let a background task die silently
        state.fail(traceback.format_exc(limit=4))


# ============================================================================ #
# S4 helpers (run in a worker thread — blocking inference + rendering)
# ============================================================================ #

def _age_sex(features: dict) -> tuple[float | None, str | None]:
    age = features.get("NACCAGE")
    try:
        age = float(age) if age is not None else None
    except (TypeError, ValueError):
        age = None
    sex = {1: "M", 2: "F", "1": "M", "2": "F"}.get(features.get("SEX"))
    return age, sex


def _run_imaging(
    plan, inp: CaseInputs, age: float | None, sex: str | None,
) -> tuple[list[tuple[ImagingCheck, Metric]], list[str], list[str]]:
    """Segment once, then derive each planned metric. Failures → UNAVAILABLE."""
    metrics: list[tuple[ImagingCheck, Metric]] = []
    tools: list[str] = []
    caveats: list[str] = []

    needs_t1 = inp.t1_path and any(c in _T1_CHECKS for c in plan.checks)
    if needs_t1:
        try:
            segmentation.segment(inp.t1_path)  # cached → derive_metric is geometry only (Inv #6)
            tools.append("segment_t1")
        except Exception as exc:
            caveats.append(f"T1 segmentation unavailable ({exc}); structural metrics "
                           "could not be computed and are reported as not assessed.")

    for check in plan.checks:
        try:
            if check is ImagingCheck.FAZEKAS:
                metric = wmh.fazekas(inp.flair_path)
                tools.append("wmh_fazekas")
            else:
                metric = segmentation.derive_metric(inp.t1_path, check, age=age, sex=sex)
                tools.append(f"derive_metric:{check.value}")
        except Exception as exc:  # belt-and-braces — no fabricated value
            metric = Metric(
                etiology=CHECK_ETIOLOGY[check], name=check.value, value=None,
                threshold=0.0, comparator="<", status=MetricStatus.UNAVAILABLE,
                caveats=[f"compute failed: {exc}"])
        metrics.append((check, metric))
    return metrics, tools, caveats


def _translate_all(
    metrics: list[tuple[ImagingCheck, Metric]], plan, inp: CaseInputs,
) -> list[FindingCard]:
    """S4d: a finding card per metric (+ explicit abstain cards) — Inv #7/#8."""
    cards: list[FindingCard] = []
    for check, metric in metrics:
        t1 = inp.t1_path if check in _T1_CHECKS else None
        cards.append(reasoner.translate(metric, check=check, t1_path=t1))

    modalities = router.available_modalities(inp.t1_path, inp.flair_path)
    for etiology in plan.abstained:
        cards.append(reasoner.abstain_card(etiology, modalities))
    return cards


# ============================================================================ #
# S6 — reasoning + self-check loop
# ============================================================================ #

async def _reason_and_self_check(state: CaseState) -> None:
    markdown = await reasoner.write_report(state.unified)
    max_iter = _max_iter()

    for attempt in range(1, max_iter + 1):
        state.set_stage(self_check_stage(attempt))
        violations = await reasoner.critique(markdown)

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
                        "returned with unresolved CDS findings.")
            return

        state.set_stage(revising_stage(attempt))
        markdown = await reasoner.revise(markdown, violations, state.unified)
