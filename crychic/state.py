"""In-memory, ephemeral case store.

A :class:`CaseState` is the single shared object a pipeline run mutates as it
advances and the status/evidence readers consume. There is intentionally no
persistence: this is a CDS *draft* workspace, and nothing here is a medical record
until a clinician signs the report outside the system.

Concurrency model
-----------------
Everything runs on one asyncio event loop. The pipeline coroutine mutates a
``CaseState`` between ``await`` points; status/evidence readers only read it.
Blocking inference is pushed to worker threads via ``asyncio.to_thread``, but the
*result* is assigned back on the loop thread, so the state object is never touched
from two threads at once.

Inputs are clinical + T1 (± FLAIR) + an optional precomputed MRI embedding fed to
the multimodal Xue model. PET/tracer input remains out of scope (§8).
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
    CrychicReport,
    Differential,
    FindingCard,
    ImagingPlan,
    UnifiedEvidence,
    XueFeatures,
)


class Stage(str, Enum):
    """The fixed spine stages (CLAUDE.md §3). The critic loop also sets dynamic
    string stages (``self_check_attempt_1``, ``revising_attempt_1``, ...).
    """

    INITIALIZED = "initialized"
    EXTRACTING = "extracting"        # S1
    SCREENING = "screening"          # S2 xue_predict
    ROUTING = "routing"              # S3 router
    IMAGING = "imaging"              # S4a/b/c
    TRANSLATING = "translating"      # S4d finding cards
    AGGREGATING = "aggregating"      # S5
    REASONING = "reasoning"          # S6
    SELF_CHECK_PASSED = "self_check_passed"
    COMPLETE = "complete"
    FAILED = "failed"


def self_check_stage(attempt: int) -> str:
    return f"self_check_attempt_{attempt}"


def revising_stage(attempt: int) -> str:
    return f"revising_attempt_{attempt}"


@dataclass
class CaseInputs:
    """What the caller handed to the pipeline.

    ``clinical`` is a raw UDS dict, a path to a record, or a free-text note (S1
    resolves it). ``mri_emb_path`` is the optional precomputed SwinUNETR embedding
    fed to the multimodal Xue model (S2). ``t1_path`` drives the independent
    structural geometry (S4); ``flair_path`` is optional and enables the WMH/Fazekas
    (VD) axis — without it that axis abstains.
    """

    clinical: dict | str
    t1_path: str | None = None
    flair_path: str | None = None
    mri_emb_path: str | None = None


@dataclass
class CaseState:
    """Mutable, per-case workspace shared between the pipeline and the readers."""

    case_id: str
    inputs: CaseInputs
    stage: str = Stage.INITIALIZED.value
    completed_tools: list[str] = field(default_factory=list)
    error: str | None = None
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None

    # Stage outputs, filled in as the pipeline advances.
    features: XueFeatures | None = None
    differential: Differential | None = None
    plan: ImagingPlan | None = None
    cards: list[FindingCard] = field(default_factory=list)
    unified: UnifiedEvidence | None = None
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

    # --- read helpers (called by the status/evidence layer) --------------- #
    def status(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "stage": self.stage,
            "completed_tools": list(self.completed_tools),
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }

    def evidence(self, fields: list[str] | None = None) -> CaseEvidence:
        """Build the ``CaseEvidence`` view; ``fields`` selects a subset.

        Selectable: ``features``, ``differential``, ``plan``, ``cards``,
        ``reconciliations``, ``conflicts``, ``report``. Reconciliations and
        conflicts are read from the unified bundle when present.
        """
        want = set(fields) if fields else None

        def take(name: str) -> bool:
            return want is None or name in want

        recon = self.unified.reconciliations if self.unified else None
        conflicts = self.unified.conflicts if self.unified else None
        return CaseEvidence(
            case_id=self.case_id,
            features=self.features if take("features") else None,
            differential=self.differential if take("differential") else None,
            plan=self.plan if take("plan") else None,
            cards=(self.cards if take("cards") else None) or None,
            reconciliations=(recon if take("reconciliations") else None),
            conflicts=(conflicts if take("conflicts") else None),
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
            while case_id in self._cases:
                case_id = f"case_{uuid.uuid4().hex[:8]}"
            state = CaseState(case_id=case_id, inputs=inputs)
            self._cases[case_id] = state
            return state

    def get(self, case_id: str) -> CaseState | None:
        return self._cases.get(case_id)

    def ids(self) -> list[str]:
        return list(self._cases)


# One store per process. The pipeline writes here; the readers read here.
STORE = CaseStore()
