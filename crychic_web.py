#!/usr/bin/env python
"""CRYCHIC web UI — a thin FastAPI front end over the in-process pipeline (S7).

Drives the same spine the MCP servers' tools wrap (clinical screen on the local
GPU, MONAI structural metrics on the local GPU, the LLM steps on the configured
Nebius/Claude endpoint), exposed as a clinical single-page app:

    GET  /                              the single-page UI (webui/index.html)
    GET  /api/cases                     list the bundled OASIS-3 demo cohort
    POST /api/run?case=OAS30209         fire the pipeline for one case -> {case_id}
    GET  /api/status/{case_id}          poll stage / completed tools / elapsed
    GET  /api/evidence/{case_id}        full structured evidence + Markdown report
    GET  /api/cards/{case_id}/{etiology}.png   the annotated key slice for a finding

The annotated finding images are rendered once during the pipeline's S4d step
(``overlay.render_overlay`` → PNG on disk); this layer just serves them. Binds
localhost only — that is the "PHI never leaves the box" guarantee (Inv #10).

Run (on the GPU box, with the LLM endpoint configured):
    NEBIUS_URL=https://api.studio.nebius.com/v1/chat/completions \
    NEBIUS_API_KEY=... NEBIUS_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    TIER1_DEVICE=cuda TIER2_DEVICE=cuda \
    uvicorn crychic_web:app --host 127.0.0.1 --port 8080
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response

from crychic import llm_client
from crychic.cases import find_emb, find_flair, find_t1, load_demo_cases
from crychic.pipeline import start_pipeline
from crychic.report import render_report_html
from crychic.state import STORE, CaseInputs

_REPO_ROOT = Path(__file__).resolve().parent
_WEBUI = _REPO_ROOT / "webui"

app = FastAPI(title="CRYCHIC", docs_url="/api/docs")


# ============================================================================ #
# Demo-cohort helpers
# ============================================================================ #

def _sex_letter(clinical: dict) -> str | None:
    return {1: "M", 2: "F"}.get(clinical.get("SEX"))


def _sex_word(clinical: dict) -> str | None:
    return {1: "Male", 2: "Female"}.get(clinical.get("SEX"))


def _as_int(v: Any) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _case_card(c) -> dict[str, Any]:
    age = c.clinical.get("NACCAGE")
    sex = _sex_letter(c.clinical)
    demo = f"{int(age)}{sex}" if (age is not None and sex) else (
        f"{int(age)}y" if age is not None else "—")
    return {
        "id": c.id,
        "subject": c.subject,
        "demo": demo,
        "age": _as_int(age),
        "sex": _sex_word(c.clinical),
        "educ": _as_int(c.clinical.get("EDUC")),
        "mmse": c.clinical.get("NACCMMSE"),
        "apoe4": c.clinical.get("NACCNE4S"),
        "true_stage": c.true_stage,
        "true_etiologies": c.true_etiologies,
        "n_clinical": len(c.clinical),
        "has_t1": find_t1(c.subject) is not None,
        "has_flair": find_flair(c.subject) is not None,
        "has_emb": find_emb(c.subject) is not None,
    }


# ============================================================================ #
# Routes
# ============================================================================ #

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_WEBUI / "index.html").read_text(encoding="utf-8")


@app.get("/api/agent")
def agent_status() -> dict[str, Any]:
    """Which LLM backend the agent steps (extract/router/reasoner) run on.

    ``online`` is False when no backend is configured — then every LLM step falls
    back to the deterministic offline template, which the UI surfaces explicitly so
    a template run is never mistaken for a live-model run.
    """
    backend, _, model = llm_client.provider_label().partition(" · ")
    return {"online": llm_client.online(), "backend": backend, "model": model or None}


@app.get("/api/cases")
def list_cases() -> dict[str, Any]:
    return {"cases": [_case_card(c) for c in load_demo_cases()]}


@app.post("/api/run")
async def run_case(case: str) -> dict[str, Any]:
    # async so start_pipeline's asyncio.create_task has a running loop.
    matches = [c for c in load_demo_cases() if case in c.id or case in c.subject]
    if not matches:
        raise HTTPException(404, f"no demo case matching {case!r}")
    c = matches[0]
    state = start_pipeline(CaseInputs(
        clinical=c.clinical,
        t1_path=find_t1(c.subject),
        flair_path=find_flair(c.subject),
        mri_emb_path=find_emb(c.subject),
    ))
    return {"case_id": state.case_id, "case": _case_card(c)}


@app.get("/api/status/{case_id}")
def status(case_id: str) -> dict[str, Any]:
    state = STORE.get(case_id)
    if state is None:
        raise HTTPException(404, "unknown case_id")
    return state.status()


@app.get("/api/evidence/{case_id}")
def evidence(case_id: str) -> JSONResponse:
    state = STORE.get(case_id)
    if state is None:
        raise HTTPException(404, "unknown case_id")
    return JSONResponse(state.evidence().model_dump(exclude_none=True))


@app.get("/api/cards/{case_id}/{etiology}.png")
def card_image(case_id: str, etiology: str) -> Response:
    """Serve the annotated key-slice PNG for one finding card (rendered at S4d)."""
    state = STORE.get(case_id)
    if state is None:
        raise HTTPException(404, "unknown case_id")
    card = next((c for c in state.cards if c.etiology == etiology), None)
    if card is None or not card.overlay_png_path:
        raise HTTPException(404, "no annotated image for this finding")
    png = Path(card.overlay_png_path)
    if not png.exists():
        raise HTTPException(404, "annotated image not found on disk")
    return Response(png.read_bytes(), media_type="image/png")


@app.get("/api/report/{case_id}.html", response_class=HTMLResponse)
def report(case_id: str) -> str:
    """The self-contained clinical report (printable to PDF), key slices embedded."""
    state = STORE.get(case_id)
    if state is None:
        raise HTTPException(404, "unknown case_id")
    if state.unified is None:
        raise HTTPException(425, "report not ready — pipeline still running")
    return render_report_html(state.unified)


def main() -> None:
    import os
    import uvicorn
    uvicorn.run(app, host=os.environ.get("CRYCHIC_HOST", "127.0.0.1"),  # Inv #10
                port=int(os.environ.get("CRYCHIC_PORT", "8080")))


if __name__ == "__main__":
    main()
