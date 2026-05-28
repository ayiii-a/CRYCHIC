#!/usr/bin/env python
"""Drive the full CRYCHIC pipeline on one bundled OASIS-3 demo case.

Exercises the real tool path end to end: Tier-1 (Xue 2024) + MONAI wholeBrainSeg
run on the local GPU, while the four LLM steps (router rationale, reasoner,
critic, reviser) go to the remote Nemotron API. It picks a case from
``data/crychic_oasis12.csv`` via the demo loader, calls the same
``start_pipeline`` the ``run_crychic_pipeline`` MCP tool uses, polls
``get_pipeline_status`` to completion, and prints ``get_case_evidence``.

A preflight ping hits Nemotron once before the run: if it fails, every LLM step
silently falls back to the offline template, so we say so loudly rather than let
you mistake a template run for an LLM run.

Usage
-----
    NEMOTRON_URL=https://integrate.api.nvidia.com/v1/chat/completions \
    NEMOTRON_MODEL=nvidia/nemotron-nano-9b-v2 \
    NEMOTRON_API_KEY=nvapi-... \
    TIER1_DEVICE=cuda TIER2_DEVICE=cuda \
    python scripts/run_one_case.py --case OAS30209

    # clinical + embedding only, skip the MONAI T1 step:
    python scripts/run_one_case.py --case OAS30209 --no-t1
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from crychic import llm
from crychic.cases import load_demo_cases
from crychic.pipeline import start_pipeline
from crychic.state import CaseInputs

_PREPRO = _REPO_ROOT / "data" / "mri_prepro"
_PET_NPY = _REPO_ROOT / "data" / "pet_npy"


def _find_t1(subject: str) -> Path | None:
    """The skull-stripped, MNI-registered T1 for this subject (MONAI input)."""
    hits = sorted(_PREPRO.glob(f"{subject}_*_stripped_MNI.nii.gz"))
    return hits[0] if hits else None


def _find_pet(subject: str) -> tuple[Path, str] | None:
    """The preprocessed amyloid-PET npy + tracer label for this subject, or None.

    Files are named ``<subject>_<TRACER>_<day>.npy`` (see
    ``scripts/preprocess_pet.py``). The tracer is whichever MYGO label the
    preprocessor canonicalized to (``AV45`` is rewritten to ``FBP``).
    """
    hits = sorted(_PET_NPY.glob(f"{subject}_*.npy"))
    if not hits:
        return None
    name = hits[0].stem            # e.g. OAS30209_PIB_d1971
    parts = name.split("_")
    tracer = parts[1] if len(parts) >= 3 else ""
    return hits[0], tracer


async def _preflight() -> None:
    """Confirm the remote Nemotron API actually answers.

    Without this, a bad URL/key just trips the per-step ``except`` and the
    pipeline runs on the deterministic template — a green run that never touched
    the LLM. We surface that instead.
    """
    if not os.environ.get("NEMOTRON_URL"):
        print("⚠  NEMOTRON_URL unset — every LLM step will use the OFFLINE "
              "TEMPLATE, not Nemotron.\n")
        return
    key_state = "with key" if llm._api_key() else "NO key"
    try:
        reply = await llm._chat(
            "You are a connectivity health check.",
            "Reply with the single word OK.",
            max_tokens=8,
        )
        print(f"✓ Nemotron reachable [{llm._model()}, {key_state}] → {reply!r}\n")
    except Exception as e:  # noqa: BLE001 — report any transport/auth failure
        print(f"✗ Nemotron call FAILED [{key_state}]: {type(e).__name__}: {e}\n"
              "  The pipeline will fall back to the offline template for all "
              "LLM steps. Fix NEMOTRON_URL / NEMOTRON_API_KEY to test the LLM.\n")


async def run(case_filter: str | None, use_t1: bool, use_pet: bool) -> int:
    cases = load_demo_cases()
    if case_filter:
        cases = [c for c in cases
                 if case_filter in c.id or case_filter in c.subject]
    if not cases:
        print(f"no case matching {case_filter!r}", file=sys.stderr)
        return 1
    case = cases[0]

    t1 = _find_t1(case.subject) if use_t1 else None
    pet_info = _find_pet(case.subject) if use_pet else None
    pet_path, pet_tracer = (pet_info if pet_info else (None, None))

    print(f"CASE {case.id}  (subject {case.subject})")
    print(f"  truth     : stage={case.true_stage}  "
          f"etiologies={case.true_etiologies or '-'}")
    print(f"  clinical  : {len(case.clinical)} UDS variables")
    print(f"  embedding : {case.embedding_path}")
    print(f"  T1 (MONAI): {t1 if t1 else '(skipped)'}")
    print(f"  PET (MYGO): {pet_path} [{pet_tracer}]" if pet_path
          else "  PET (MYGO): (skipped)")
    print()

    await _preflight()

    inputs = CaseInputs(
        clinical=case.clinical,
        t1_path=str(t1) if t1 else None,
        pet_path=str(pet_path) if pet_path else None,
        tracer=pet_tracer,
        mri_embedding_path=str(case.embedding_path) if case.embedding_path else None,
    )
    state = start_pipeline(inputs)
    print(f"started {state.case_id}; polling get_pipeline_status ...\n")

    last = None
    while not state.is_terminal:
        st = state.status()
        sig = (st["stage"], tuple(st["completed_tools"]))
        if sig != last:
            print(f"  [{st['elapsed_seconds']:>5.1f}s] {st['stage']:<22} "
                  f"tools={st['completed_tools']}")
            last = sig
        await asyncio.sleep(0.2)
    st = state.status()
    print(f"  [{st['elapsed_seconds']:>5.1f}s] {state.stage}  (done)\n")

    if state.error:
        print("PIPELINE FAILED:\n" + state.error)
        return 1

    ev = state.evidence()
    t1r = ev.tier1
    print("── Tier-1 ──────────────────────────────────────────────")
    print(f"  stage_top={t1r.stage_top}  etiology_top={t1r.etiology_top}  "
          f"imaging_used={t1r.imaging_used}")
    print(f"  P(AD)={t1r.p_ad:.2f}  P(MCI)={t1r.p_mci:.2f}  P(VD)={t1r.p_vd:.2f}")
    print(f"  routing : {[t.value for t in ev.routing.selected_tools]}")
    print(f"  pattern : #{ev.pattern.pattern_id} {ev.pattern.name}")
    for c in ev.conflicts or []:
        print(f"  conflict: [{c.severity.value}] {c.name}")

    if ev.tier2 and ev.tier2.centiloid:
        cl = ev.tier2.centiloid
        print("\n── Tier-2 Centiloid ────────────────────────────────────")
        print(f"  source={cl.source.value}  tracer={cl.tracer}  "
              f"centiloid={cl.centiloid}  positive={cl.positive} "
              f"(≥{cl.threshold})")
        for c in cl.caveats:
            print(f"  caveat: {c}")

    rep = ev.report
    print(f"\n  self_check_passed={rep.self_check_passed}  "
          f"iterations={rep.iterations}")
    if rep.warning:
        print(f"  WARNING: {rep.warning}")
    if rep.remaining_violations:
        for r in rep.remaining_violations:
            print(f"    - {r}")

    print("\n══ REPORT ═══════════════════════════════════════════════\n")
    print(rep.markdown)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--case", default=None,
                    help="substring of case id/subject (e.g. OAS30209). "
                         "Default: first case in the cohort.")
    ap.add_argument("--no-t1", action="store_true",
                    help="skip the MONAI T1 step (clinical + synthetic imaging).")
    ap.add_argument("--no-pet", action="store_true",
                    help="skip the MYGO PET step (centiloid falls back to synthetic).")
    args = ap.parse_args()
    return asyncio.run(run(args.case, use_t1=not args.no_t1, use_pet=not args.no_pet))


if __name__ == "__main__":
    raise SystemExit(main())
