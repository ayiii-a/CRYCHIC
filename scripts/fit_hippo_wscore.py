#!/usr/bin/env python
"""Fit the hippocampal z-score coefficients from a cognitively-normal cohort.

This produces the *correct*, same-pipeline normative model for the hippocampal
biomarker (CLAUDE.md §4 credibility anchor). It segments cognitively-normal
reference subjects with the SAME MONAI bundle the pipeline uses, then OLS-fits

    V_hippo_total_mm3 ~ intercept + b_age·age + b_sex_male·sex_male + b_tiv·TIV

and writes the coefficients + residual SD to ``crychic/norms/hippo_wscore.json``.
At inference, ``geometry.hippocampus_z()`` turns that into a z-score
``z = (V - V_pred) / residual_SD`` (atrophy at z < -1.5) — the BrainChart /
Potvin 2016 normative-modelling approach, but calibrated to THIS segmenter so
there is no cross-method bias.

Manifest CSV columns (one cognitively-normal subject per row):
    t1_path   path to the skull-stripped, MNI-registered T1 (.nii.gz)
    age       age in years
    sex       M/F or NACC 1/2

Usage
-----
    TIER2_DEVICE=cuda python scripts/fit_hippo_wscore.py --manifest cn_cohort.csv

The OLS ``fit_wscore`` is pure and unit-tested offline; only ``gather`` needs torch.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_OUT = _REPO_ROOT / "crychic" / "norms" / "hippo_wscore.json"
_REFERENCE = ("Hippocampal z-score (age/sex/TIV-adjusted); coefficients fit by OLS on "
              "cognitively-normal subjects segmented by this MONAI wholeBrainSeg bundle. "
              "Method per BrainChart (Bethlehem 2022) / Potvin 2016.")


@dataclass
class Record:
    hippo_mm3: float
    tiv_mm3: float
    age: float | None = None
    sex_male: float | None = None      # 1.0 male, 0.0 female


def _sex_male(raw) -> float | None:
    return {"M": 1.0, "F": 0.0, "1": 1.0, "2": 0.0,
            "MALE": 1.0, "FEMALE": 0.0}.get(str(raw).strip().upper())


# ============================================================================ #
# Pure OLS fit — no torch, unit-testable offline
# ============================================================================ #

def fit_wscore(records: list[Record], *, source: str = "cohort") -> dict:
    """OLS-fit the z-score coefficients from reference records (pure; no I/O)."""
    import numpy as np

    rows = [r for r in records
            if r.hippo_mm3 and r.tiv_mm3 and r.age is not None and r.sex_male is not None]
    if len(rows) < 5:
        raise ValueError(f"need ≥5 complete (hippo, tiv, age, sex) subjects to fit; "
                         f"got {len(rows)}")

    X = np.array([[1.0, r.age, r.sex_male, r.tiv_mm3] for r in rows], dtype=float)
    y = np.array([r.hippo_mm3 for r in rows], dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(rows) - X.shape[1])
    resid_sd = float(np.sqrt(float(resid @ resid) / dof))

    return {
        "model": "hippocampus_wscore_v1",
        "formula": "V_hippo_total_mm3 ~ intercept + b_age*age_years + b_sex_male*sex_male + b_tiv*tiv_mm3",
        "intercept": round(float(beta[0]), 4),
        "b_age": round(float(beta[1]), 4),
        "b_sex_male": round(float(beta[2]), 4),
        "b_tiv": round(float(beta[3]), 8),
        "residual_sd": round(resid_sd, 2),
        "age_mean": round(float(np.mean([r.age for r in rows])), 2),
        "sex_male_mean": round(float(np.mean([r.sex_male for r in rows])), 4),
        "tiv_mean": round(float(np.mean([r.tiv_mm3 for r in rows])), 1),
        "n": len(rows),
        "source": source,
        "threshold": -1.5,
        "reference": _REFERENCE,
    }


# ============================================================================ #
# Cohort segmentation — needs torch + MONAI
# ============================================================================ #

def gather(manifest: Path) -> list[Record]:
    """Segment every T1 in the manifest with the SAME bundle → reference records."""
    from crychic import segmentation

    records: list[Record] = []
    with manifest.open() as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            t1 = (row.get("t1_path") or "").strip()
            if not t1:
                continue
            try:
                summ = segmentation.segment(t1)
            except Exception as exc:
                print(f"  skip row {i} ({t1}): segmentation failed — {exc}", file=sys.stderr)
                continue
            try:
                age = float(row["age"]) if row.get("age") not in (None, "") else None
            except ValueError:
                age = None
            rec = Record(hippo_mm3=float(summ.get("hippocampus_total_mm3") or 0),
                         tiv_mm3=float(summ.get("tiv_mm3") or 0),
                         age=age, sex_male=_sex_male(row.get("sex")))
            if not rec.hippo_mm3 or not rec.tiv_mm3:
                print(f"  skip row {i} ({t1}): no hippocampus/TIV measured", file=sys.stderr)
                continue
            records.append(rec)
            print(f"  [{i}] hippo={rec.hippo_mm3:.0f} tiv={rec.tiv_mm3:.0f} "
                  f"age={rec.age} sex_male={rec.sex_male}")
    return records


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path,
                    help="CSV of cognitively-normal reference subjects (t1_path,age,sex).")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                    help=f"Output coefficients JSON (default: {_DEFAULT_OUT}).")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    print(f"Segmenting CN cohort from {args.manifest} (same MONAI bundle) ...")
    records = gather(args.manifest)
    try:
        coeffs = fit_wscore(records)
    except ValueError as exc:
        print(f"fit failed: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(coeffs, indent=2) + "\n")
    print(f"\nWrote z-score coefficients (n={coeffs['n']}) → {args.out}")
    for k in ("intercept", "b_age", "b_sex_male", "b_tiv", "residual_sd"):
        print(f"  {k:12} {coeffs[k]}")
    if coeffs["n"] < 20:
        print(f"  ⚠ n={coeffs['n']} (<20): geometry.hippocampus_z() will still flag "
              "this as not-yet-validated; gather more CN subjects.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
