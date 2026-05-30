"""Loader for the bundled OASIS-3 demo cohort.

Demo-data glue (``data/crychic_oasis12.csv``), not part of the pipeline contracts.
For each record it strips the trailing ground-truth label columns from the model
inputs and exposes the held-out labels for display.

It also centralizes per-subject file discovery (T1, FLAIR, MRI embedding) so the
web UI and the CLI resolve the same paths from one place: :func:`find_t1`,
:func:`find_flair`, :func:`find_emb`.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"

# The CSV's trailing 13 columns are held-out ground truth, not model inputs.
# Pandas renames the TBI *etiology* label to "TBI.1" to avoid clashing with the
# TBI *history* clinical feature; we strip the former and keep the latter.
_STAGE_LABELS = ("NC", "MCI", "DE")
_ETIOLOGY_LABELS = ("AD", "LBD", "VD", "PRD", "FTD", "NPH", "SEF", "PSY", "TBI", "ODE")
_LABEL_COLS = {"NC", "MCI", "DE", "AD", "LBD", "VD", "PRD", "FTD", "NPH",
               "SEF", "PSY", "TBI.1", "ODE"}
_CSV_TO_LABEL = {"TBI.1": "TBI"}  # canonicalize the disambiguated column name
_OAS_RE = re.compile(r"(OAS\d+)")


@dataclass
class DemoCase:
    id: str                       # e.g. "A1_OAS30073"
    subject: str                  # e.g. "OAS30073"
    clinical: dict                # UDS variables fed to the model (labels removed)
    labels: dict[str, int]        # ground truth, canonical names {NC, AD, ...}

    @property
    def true_stage(self) -> str | None:
        on = [l for l in _STAGE_LABELS if self.labels.get(l)]
        return on[0] if on else None

    @property
    def true_etiologies(self) -> list[str]:
        return [l for l in _ETIOLOGY_LABELS if self.labels.get(l)]


def _data_dir() -> Path:
    return Path(os.environ.get("CRYCHIC_DATA_DIR", _DATA_DIR)).expanduser()


# --- per-subject imaging file discovery (one source of truth) ---------------- #

def _first(glob_dir: Path, pattern: str) -> str | None:
    hits = sorted(glob_dir.glob(pattern))
    return str(hits[0]) if hits else None


def find_t1(subject: str) -> str | None:
    """The skull-stripped, MNI-registered T1 for a subject (the MONAI seg input)."""
    return _first(_data_dir() / "mri_prepro", f"{subject}_*_stripped_MNI.nii.gz")


def find_flair(subject: str) -> str | None:
    """A FLAIR coregistered to the T1 grid, if present (enables the VD/Fazekas axis).

    Produced by ``scripts/coreg_flair.py``; absent for the bundled T1-only cohort.
    """
    return _first(_data_dir() / "mri_prepro", f"{subject}_*FLAIR*.nii.gz")


def find_emb(subject: str) -> str | None:
    """The precomputed SwinUNETR embedding for the multimodal Xue model, if present."""
    return _first(_data_dir() / "mri_emb", f"{subject}_*_emb.npy")


def _clean(value):
    """Drop NaNs; pandas reads sparse UDS columns as floats."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def load_demo_cases(csv_path: str | Path | None = None) -> list[DemoCase]:
    """Load every case in the demo cohort as a :class:`DemoCase` (clinical-only)."""
    data_dir = _data_dir()
    csv_path = Path(csv_path) if csv_path else data_dir / "crychic_oasis12.csv"

    df = pd.read_csv(csv_path)
    cases: list[DemoCase] = []
    for _, row in df.iterrows():
        rid = str(row["ID"])
        m = _OAS_RE.search(rid)
        subject = m.group(1) if m else rid

        clinical, labels = {}, {}
        for col, val in row.items():
            val = _clean(val)
            if val is None:
                continue
            if col in _LABEL_COLS:
                labels[_CSV_TO_LABEL.get(col, col)] = int(val)
            else:
                clinical[col] = val

        cases.append(DemoCase(id=rid, subject=subject, clinical=clinical, labels=labels))
    return cases
