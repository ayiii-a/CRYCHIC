#!/usr/bin/env python
"""Inspect the bundled OASIS-3 demo cohort through the clinical differential.

Runs the Xue 2024 model on each case in ``data/crychic_oasis12.csv`` — clinical
features ONLY (Inv #1; no MRI is ever fed to the model) — and prints the predicted
stage/etiology against the held-out truth, with a simple top-1 tally.

Examples
--------
    python scripts/inspect_cases.py                 # whole cohort
    python scripts/inspect_cases.py --case OAS30209 # one case
    python scripts/inspect_cases.py --no-explain    # skip the attribution heatmap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from crychic import xue
from crychic.cases import DemoCase, load_demo_cases


def _fmt_etio(etios: list[str]) -> str:
    return ",".join(etios) if etios else "-"


def inspect(cases: list[DemoCase], explain: bool) -> None:
    hdr = (f"{'CASE':<14}{'TRUTH (stage/etio)':<26}"
           f"{'PRED stage':<11}{'PRED etio':<10}"
           f"{'p_AD':>6}{'p_MCI':>7}{'p_VD':>6}  RESULT")
    print(hdr)
    print("-" * len(hdr))

    stage_hits = etio_hits = etio_evaluable = 0
    for c in cases:
        r = xue.screen(c.clinical, explain=explain)

        truth = f"{c.true_stage or '-'} / {_fmt_etio(c.true_etiologies)}"
        stage_ok = (c.true_stage == r.stage_top)
        stage_hits += stage_ok
        notes = ["stage" + ("✓" if stage_ok else "✗")]
        if c.true_etiologies:
            etio_evaluable += 1
            etio_ok = r.etiology_top in c.true_etiologies
            etio_hits += etio_ok
            notes.append("etio" + ("✓" if etio_ok else "✗"))

        print(f"{c.id:<14}{truth:<26}{r.stage_top:<11}{r.etiology_top:<10}"
              f"{r.p_ad:>6.2f}{r.p_mci:>7.2f}{r.p_vd:>6.2f}  {' '.join(notes)}")

    n = len(cases)
    print("-" * len(hdr))
    line = f"stage top-1: {stage_hits}/{n}"
    if etio_evaluable:
        line += f"   |   etiology top-1 (on {etio_evaluable} non-NC): {etio_hits}/{etio_evaluable}"
    print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-explain", action="store_true",
                    help="skip the attribution heatmap (faster)")
    ap.add_argument("--case", default=None,
                    help="substring filter on case id (e.g. OAS30209)")
    args = ap.parse_args()

    cases = load_demo_cases()
    if args.case:
        cases = [c for c in cases if args.case in c.id]
        if not cases:
            print(f"no case matching {args.case!r}", file=sys.stderr)
            return 1

    print(f"Loaded {len(cases)} cases  |  clinical features only (no imaging)\n")
    xue.warmup()
    inspect(cases, explain=not args.no_explain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
