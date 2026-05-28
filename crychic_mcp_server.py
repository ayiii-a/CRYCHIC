#!/usr/bin/env python
"""CRYCHIC MCP server — 3 tools over the dementia CDS pipeline.

Exposes the whole pipeline (Tier-1 screening, rule-based routing, parallel
imaging, aggregation, and the LLM reasoner + self-check loop) behind three clean
tools that an orchestrator can call:

    run_crychic_pipeline   fire the pipeline; returns a case_id immediately
    get_pipeline_status    poll the current stage / completed tools / elapsed
    get_case_evidence      retrieve structured intermediate evidence

The LLM is called *from inside* the pipeline, so the orchestrator never has to
do tool-calling on a small model — see CLAUDE.md. Transport is selected by
``MCP_TRANSPORT`` (``stdio`` default, or ``sse``).

Run:
    NEMOTRON_URL=http://localhost:8000/v1/chat/completions python crychic_mcp_server.py
    npx @modelcontextprotocol/inspector python crychic_mcp_server.py
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from crychic.pipeline import start_pipeline
from crychic.state import STORE, CaseInputs

mcp = FastMCP("crychic")


@mcp.tool()
def run_crychic_pipeline(
    clinical: dict[str, Any] | None = None,
    clinical_text: str | None = None,
    t1_path: str | None = None,
    pet_path: str | None = None,
    tracer: str | None = None,
    mri_embedding_path: str | None = None,
) -> dict[str, Any]:
    """Start the full CRYCHIC pipeline for one case (runs async in the background).

    Provide ``clinical`` as a dict of raw UDS variables (preferred — e.g.
    ``{"NACCAGE": 78, "NACCMMSE": 22, "SEX": 0, "NACCNE4S": 1}``); ``clinical_text``
    is accepted for compatibility but only used if it is a path to a JSON/CSV
    record (free prose is not parsed into UDS variables in v0.3). ``t1_path`` is a
    structural T1 ``.nii.gz`` (drives MONAI wholeBrainSeg and is the MONAI
    trigger). ``pet_path`` + ``tracer`` feed amyloid quantification; both are
    optional, since amyloid degrades to a surfaced limitation when absent.
    ``mri_embedding_path`` is an optional precomputed SwinUNETR ``.npy`` to speed
    up Tier-1.

    Returns ``{case_id, status, message}``; poll ``get_pipeline_status`` for
    progress and ``get_case_evidence`` for results.
    """
    clinical_input: dict | str = clinical if clinical is not None else (clinical_text or {})
    inputs = CaseInputs(
        clinical=clinical_input,
        t1_path=t1_path,
        pet_path=pet_path,
        tracer=tracer,
        mri_embedding_path=mri_embedding_path,
    )
    state = start_pipeline(inputs)
    return {
        "case_id": state.case_id,
        "status": "started",
        "message": "Pipeline running async. Poll get_pipeline_status.",
    }


@mcp.tool()
def get_pipeline_status(case_id: str) -> dict[str, Any]:
    """Poll one case's current execution stage.

    Returns ``{case_id, stage, completed_tools, elapsed_seconds, error}``. The
    ``stage`` walks: initialized → screening → routing → imaging → aggregating →
    reasoning → self_check_attempt_N (↔ revising_attempt_N) → self_check_passed →
    complete | failed.
    """
    state = STORE.get(case_id)
    if state is None:
        return {"case_id": case_id, "error": "unknown case_id", "stage": None}
    return state.status()


@mcp.tool()
def get_case_evidence(
    case_id: str, fields: list[str] | None = None
) -> dict[str, Any]:
    """Retrieve structured intermediate evidence for a case.

    ``fields`` selects a subset of ``tier1``, ``routing``, ``tier2``, ``pattern``,
    ``conflicts``, ``report``; omit it to get everything available so far. Useful
    for follow-up questions ("what was the Centiloid value?") with no pipeline
    re-run. Note evidence fills in stage by stage — poll status for completion.
    """
    state = STORE.get(case_id)
    if state is None:
        return {"case_id": case_id, "error": "unknown case_id"}
    evidence = state.evidence(fields)
    # exclude_none keeps the payload tight; the report markdown is the big field.
    return evidence.model_dump(exclude_none=True)


def main() -> None:
    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()
    if transport == "sse":
        mcp.settings.port = int(os.environ.get("MCP_SSE_PORT", "3000"))
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
