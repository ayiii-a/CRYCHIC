#!/usr/bin/env python
"""Inspect the bundled OASIS-3 demo cohort through Tier-1 screening.

Runs the Xue 2024 model on each case in ``data/crychic_oasis12.csv``, using the
matching precomputed MRI embedding when available. MRI is **optional**: pass
``--no-mri`` to force the clinical-only screen, or ``--compare`` to see both
side by side and how much the uploaded scan shifts the prediction.

Examples
--------
    python scripts/inspect_cases.py                 # with MRI embeddings
    python scripts/inspect_cases.py --no-mri        # clinical-only
    python scripts/inspect_cases.py --compare       # clinical-only vs +MRI
    python scripts/inspect_cases.py --case OAS30209 # one case
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from crychic import tier1_screening as t1
from crychic.cases import DemoCase, load_demo_cases


def _fmt_etio(etios: list[str]) -> str:
    return ",".join(etios) if etios else "-"


def _mri_influence(result) -> str:
    """Largest |Δ| label from the heatmap's 'MRI (img)' row, if present."""
    h = result.heatmap
    if not h or "MRI (img)" not in h.rows:
        return ""
    row = h.values[h.rows.index("MRI (img)")]
    label, delta = max(zip(h.columns, row), key=lambda kv: abs(kv[1]))
    sign = "+" if delta >= 0 else ""
    return f"{label} {sign}{delta:.3f}"


def inspect(cases: list[DemoCase], use_mri: bool, explain: bool) -> None:
    hdr = (f"{'CASE':<14}{'MRI':<5}{'TRUTH (stage/etio)':<26}"
           f"{'PRED stage':<11}{'PRED etio':<10}"
           f"{'p_AD':>6}{'p_MCI':>7}{'p_VD':>6}  RESULT")
    print(hdr)
    print("-" * len(hdr))

    stage_hits = etio_hits = etio_evaluable = 0
    for c in cases:
        mri = c.embedding_path if (use_mri and c.has_mri) else None
        r = t1.screen(c.clinical, mri=mri, explain=explain)

        truth = f"{c.true_stage or '-'} / {_fmt_etio(c.true_etiologies)}"
        stage_ok = (c.true_stage == r.stage_top)
        stage_hits += stage_ok
        notes = ["stage" + ("✓" if stage_ok else "✗")]
        if c.true_etiologies:
            etio_evaluable += 1
            etio_ok = r.etiology_top in c.true_etiologies
            etio_hits += etio_ok
            notes.append("etio" + ("✓" if etio_ok else "✗"))

        print(f"{c.id:<14}{('yes' if mri else 'no'):<5}{truth:<26}"
              f"{r.stage_top:<11}{r.etiology_top:<10}"
              f"{r.p_ad:>6.2f}{r.p_mci:>7.2f}{r.p_vd:>6.2f}  {' '.join(notes)}")

    n = len(cases)
    print("-" * len(hdr))
    line = f"stage top-1: {stage_hits}/{n}"
    if etio_evaluable:
        line += f"   |   etiology top-1 (on {etio_evaluable} non-NC): {etio_hits}/{etio_evaluable}"
    print(line)


def compare(cases: list[DemoCase], explain: bool) -> None:
    hdr = (f"{'CASE':<14}{'TRUTH':<14}"
           f"{'clinical-only':<22}{'+MRI':<22}{'top MRI shift':<16}")
    print(hdr)
    print("-" * len(hdr))
    for c in cases:
        base = t1.screen(c.clinical, mri=None, explain=False)
        b = f"{base.stage_top}/{base.etiology_top}"
        if c.has_mri:
            withmri = t1.screen(c.clinical, mri=c.embedding_path, explain=explain)
            w = f"{withmri.stage_top}/{withmri.etiology_top}"
            shift = _mri_influence(withmri) if explain else ""
        else:
            w, shift = "(no embedding)", ""
        truth = f"{c.true_stage or '-'}/{_fmt_etio(c.true_etiologies)}"
        print(f"{c.id:<14}{truth:<14}{b:<22}{w:<22}{shift:<16}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-mri", action="store_true",
                    help="force clinical-only (ignore embeddings)")
    ap.add_argument("--compare", action="store_true",
                    help="show clinical-only vs +MRI for each case")
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

    n_with_mri = sum(c.has_mri for c in cases)
    print(f"Loaded {len(cases)} cases  |  {n_with_mri} have an MRI embedding "
          f"(MRI is optional)\n")

    t1.warmup()
    if args.compare:
        compare(cases, explain=not args.no_explain)
    else:
        inspect(cases, use_mri=not args.no_mri, explain=not args.no_explain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
