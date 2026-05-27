"""In-memory, ephemeral case store.

A :class:`CaseState` is the single shared object a pipeline run mutates as it
advances and the MCP status/evidence tools read from. There is intentionally no
persistence: this is a CDS *draft* workspace, and nothing here is a medical
record until a clinician signs the report outside the system.

Concurrency model
-----------------
Everything runs on one asyncio event loop. The pipeline coroutine mutates a
``CaseState`` between ``await`` points; the ``get_pipeline_status`` /
``get_case_evidence`` handlers only read it. Blocking inference is pushed to
worker threads via :func:`asyncio.to_thread`, but the *result* is assigned back
on the loop thread, so the state object itself is never touched from two threads
at once. Attribute writes are plain and need no lock; the store's ``create`` is
guarded only to keep id generation tidy.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any

from .schemas import (
    CaseEvidence,
    ClinicalPattern,
    Conflict,
    CrychicReport,
    RoutingDecision,
    Tier1Result,
    Tier2Result,
)


class Stage(str, Enum):
    """The fixed stages. The critic loop also sets dynamic string stages
    (``self_check_attempt_1``, ``revising_attempt_1``, ...) via :meth:`set_stage`.
    """

    INITIALIZED = "initialized"
    SCREENING = "screening"
    ROUTING = "routing"
    IMAGING = "imaging"
    AGGREGATING = "aggregating"
    REASONING = "reasoning"
    SELF_CHECK_PASSED = "self_check_passed"
    COMPLETE = "complete"
    FAILED = "failed"


def self_check_stage(attempt: int) -> str:
    return f"self_check_attempt_{attempt}"


def revising_stage(attempt: int) -> str:
    return f"revising_attempt_{attempt}"


@dataclass
class CaseInputs:
    """What the caller handed to ``run_crychic_pipeline``.

    ``clinical`` is the raw UDS dict (or free text) for Tier-1; ``pet_path`` is
    optional because the bundled demo cohort is T1-only — Centiloid degrades to
    a surfaced limitation when no PET is supplied.
    """

    clinical: dict | str
    t1_path: str | None = None
    pet_path: str | None = None
    tracer: str | None = None
    mri_embedding_path: str | None = None  # precomputed SwinUNETR .npy for Tier-1


@dataclass
class CaseState:
    """Mutable, per-case workspace shared between the pipeline and the tools."""

    case_id: str
    inputs: CaseInputs
    stage: str = Stage.INITIALIZED.value
    completed_tools: list[str] = field(default_factory=list)
    error: str | None = None
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None

    # Stage outputs, filled in as the pipeline advances.
    tier1: Tier1Result | None = None
    routing: RoutingDecision | None = None
    tier2: Tier2Result | None = None
    pattern: ClinicalPattern | None = None
    conflicts: list[Conflict] = field(default_factory=list)
    report: CrychicReport | None = None

    # --- mutation helpers (called by the pipeline) ------------------------- #
    def set_stage(self, stage: Stage | str) -> None:
        self.stage = stage.value if isinstance(stage, Stage) else stage

    def mark_tool_done(self, tool: str) -> None:
        if tool not in self.completed_tools:
            self.completed_tools.append(tool)

    def fail(self, error: str) -> None:
        self.error = error
        self.stage = Stage.FAILED.value
        self.finished_at = time.monotonic()

    def complete(self) -> None:
        self.stage = Stage.COMPLETE.value
        self.finished_at = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return round(end - self.started_at, 1)

    @property
    def is_terminal(self) -> bool:
        return self.stage in (Stage.COMPLETE.value, Stage.FAILED.value)

    # --- read helpers (called by the MCP tools) --------------------------- #
    def status(self) -> dict[str, Any]:
        """Payload for ``get_pipeline_status``."""
        return {
            "case_id": self.case_id,
            "stage": self.stage,
            "completed_tools": list(self.completed_tools),
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }

    def evidence(self, fields: list[str] | None = None) -> CaseEvidence:
        """Build the ``CaseEvidence`` view for ``get_case_evidence``.

        ``fields`` selects a subset (``tier1``, ``routing``, ``tier2``,
        ``pattern``, ``conflicts``, ``report``); ``None`` returns everything
        available so far. Unknown field names are ignored.
        """
        want = set(fields) if fields else None

        def take(name: str) -> bool:
            return want is None or name in want

        return CaseEvidence(
            case_id=self.case_id,
            tier1=self.tier1 if take("tier1") else None,
            routing=self.routing if take("routing") else None,
            tier2=self.tier2 if take("tier2") else None,
            pattern=self.pattern if take("pattern") else None,
            conflicts=(self.conflicts if take("conflicts") else None),
            report=self.report if take("report") else None,
        )


class CaseStore:
    """Process-lifetime registry of cases, keyed by ``case_id``."""

    def __init__(self) -> None:
        self._cases: dict[str, CaseState] = {}
        self._lock = Lock()

    def create(self, inputs: CaseInputs) -> CaseState:
        with self._lock:
            case_id = f"case_{uuid.uuid4().hex[:8]}"
            while case_id in self._cases:  # vanishingly unlikely; keep ids unique
                case_id = f"case_{uuid.uuid4().hex[:8]}"
            state = CaseState(case_id=case_id, inputs=inputs)
            self._cases[case_id] = state
            return state

    def get(self, case_id: str) -> CaseState | None:
        return self._cases.get(case_id)

    def ids(self) -> list[str]:
        return list(self._cases)


# One store per server process. The pipeline writes here; the tools read here.
STORE = CaseStore()
