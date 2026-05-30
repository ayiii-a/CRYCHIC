#!/usr/bin/env python
"""Drive the full CRYCHIC spine on one bundled OASIS-3 demo case.

Exercises the real path end to end: the clinical differential (Xue 2024,
clinical features only) + MONAI wholeBrainSeg structural metrics on the local GPU,
while the LLM steps (S1 extract, S3 router rationale, S6 reasoner + self-check) go
to the configured Nemotron/Claude endpoint. It picks a case from
``data/crychic_oasis12.csv``, calls the same ``start_pipeline`` the spine uses,
polls to completion, and prints the differential, the imaging plan, the finding
cards, the reconciliation, and the report.

A preflight ping hits the LLM once: if it fails, every LLM step silently falls back
to the offline template, so we say so loudly rather than let you mistake a template
run for an LLM run.

Usage
-----
    NEMOTRON_URL=https://integrate.api.nvidia.com/v1/chat/completions \
    NEMOTRON_MODEL=nvidia/nemotron-nano-9b-v2 NEMOTRON_API_KEY=nvapi-... \
    TIER1_DEVICE=cuda TIER2_DEVICE=cuda \
    python scripts/run_one_case.py --case OAS30209

    python scripts/run_one_case.py --case OAS30209 --no-t1   # clinical-only
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

from crychic import llm_client
from crychic.cases import find_emb, find_flair, find_t1, load_demo_cases
from crychic.pipeline import start_pipeline
from crychic.report import render_report_html
from crychic.state import CaseInputs


async def _preflight() -> None:
    """Confirm the LLM endpoint actually answers (else everything runs offline)."""
    if not llm_client.online():
        print("⚠  No LLM backend (set ANTHROPIC_API_KEY for Claude, or NEMOTRON_URL) — "
              "every LLM step (extract/route/reason) will use the OFFLINE TEMPLATE.\n")
        return
    try:
        reply = await llm_client.chat("You are a connectivity health check.",
                                      "Reply with the single word OK.", max_tokens=8)
        print(f"✓ LLM reachable [{llm_client.provider_label()}] → {reply!r}\n")
    except Exception as e:  # noqa: BLE001
        print(f"✗ LLM call FAILED [{key_state}]: {type(e).__name__}: {e}\n"
              "  The pipeline will fall back to the offline template for all LLM "
              "steps. Fix NEMOTRON_URL / NEMOTRON_API_KEY to test the LLM.\n")


async def run(case_filter: str | None, use_t1: bool, report: str | None = None) -> int:
    cases = load_demo_cases()
    if case_filter:
        cases = [c for c in cases if case_filter in c.id or case_filter in c.subject]
    if not cases:
        print(f"no case matching {case_filter!r}", file=sys.stderr)
        return 1
    case = cases[0]

    t1 = find_t1(case.subject) if use_t1 else None
    flair = find_flair(case.subject) if use_t1 else None
    emb = find_emb(case.subject)  # multimodal Xue input (independent of --t1)

    print(f"CASE {case.id}  (subject {case.subject})")
    print(f"  truth     : stage={case.true_stage}  etiologies={case.true_etiologies or '-'}")
    print(f"  clinical  : {len(case.clinical)} UDS variables")
    print(f"  MRI emb   : {emb if emb else '(none — clinical-only Xue)'}")
    print(f"  T1 (MONAI): {t1 if t1 else '(skipped)'}")
    print(f"  FLAIR (VD): {flair if flair else '(none — VD axis abstains)'}\n")

    await _preflight()

    inputs = CaseInputs(clinical=case.clinical, t1_path=t1, flair_path=flair,
                        mri_emb_path=emb)
    state = start_pipeline(inputs)
    print(f"started {state.case_id}; polling status ...\n")

    last = None
    while not state.is_terminal:
        st = state.status()
        sig = (st["stage"], tuple(st["completed_tools"]))
        if sig != last:
            print(f"  [{st['elapsed_seconds']:>5.1f}s] {st['stage']:<22} "
                  f"tools={st['completed_tools']}")
            last = sig
        await asyncio.sleep(0.2)
    print(f"  [{state.status()['elapsed_seconds']:>5.1f}s] {state.stage}  (done)\n")

    if state.error:
        print("PIPELINE FAILED:\n" + state.error)
        return 1

    ev = state.evidence()
    d = ev.differential
    axis = "clinical + MRI" if d.imaging_used else "clinical only"
    print(f"── Differential ({axis}) ───────────────────────")
    print(f"  stage_top={d.stage_top}  etiology_top={d.etiology_top}")
    print(f"  P(AD)={d.p_ad:.2f}  P(MCI)={d.p_mci:.2f}  P(VD)={d.p_vd:.2f}  "
          f"P(FTD)={d.p('FTD'):.2f}  P(NPH)={d.p('NPH'):.2f}")
    print(f"  plan      : checks={[c.value for c in ev.plan.checks]} "
          f"abstained={ev.plan.abstained}")

    print("\n── Finding cards ──────────────────────────────────────")
    for c in ev.cards or []:
        print(f"  [{c.polarity:10}] {c.title}")
        print(f"               {c.sentence}")

    print("\n── Reconciliation ─────────────────────────────────────")
    for r in ev.reconciliations or []:
        print(f"  {r.etiology:4} {r.recon.value}")
    for c in ev.conflicts or []:
        print(f"  conflict [{c.severity.value}] {c.name}")

    rep = ev.report
    print(f"\n  self_check_passed={rep.self_check_passed}  iterations={rep.iterations}")
    if rep.warning:
        print(f"  WARNING: {rep.warning}")

    print("\n══ REPORT ═══════════════════════════════════════════════\n")
    print(rep.markdown)

    if report is not None and state.unified is not None:
        out = f"report_{case.subject}.html" if report == "<auto>" else report
        Path(out).write_text(render_report_html(state.unified), encoding="utf-8")
        n_img = sum(1 for c in (ev.cards or []) if c.overlay_png_path)
        print(f"\n📄 wrote self-contained HTML report ({n_img} embedded slice(s)) → {out}"
              "\n   open it in a browser, then Print → Save as PDF.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default=None,
                    help="substring of case id/subject (e.g. OAS30209). "
                         "Default: first case in the cohort.")
    ap.add_argument("--no-t1", action="store_true",
                    help="skip the MONAI T1 step (clinical-only; structural axes abstain).")
    ap.add_argument("--report", nargs="?", const="<auto>", default=None, metavar="PATH",
                    help="write the self-contained HTML report (key slices embedded); "
                         "optional path, default report_<subject>.html")
    args = ap.parse_args()
    return asyncio.run(run(args.case, use_t1=not args.no_t1, report=args.report))


if __name__ == "__main__":
    raise SystemExit(main())
