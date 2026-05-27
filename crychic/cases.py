"""Loader for the bundled OASIS-3 demo cohort.

Demo-data glue (``data/crychic_oasis12.csv`` + ``data/mri_emb/*.npy``), not part
of the pipeline contracts. For each record it:

* strips the trailing ground-truth label columns from the model inputs,
* maps the case to its optional precomputed SwinUNETR embedding by subject id.

The MRI embedding is **optional** throughout: a case with no matching ``.npy``
still yields a valid clinical-only case.
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
    embedding_path: Path | None   # precomputed SwinUNETR .npy, or None
    labels: dict[str, int]        # ground truth, canonical names {NC, AD, ...}

    @property
    def has_mri(self) -> bool:
        return self.embedding_path is not None

    @property
    def true_stage(self) -> str | None:
        on = [l for l in _STAGE_LABELS if self.labels.get(l)]
        return on[0] if on else None

    @property
    def true_etiologies(self) -> list[str]:
        return [l for l in _ETIOLOGY_LABELS if self.labels.get(l)]


def _data_dir() -> Path:
    return Path(os.environ.get("CRYCHIC_DATA_DIR", _DATA_DIR)).expanduser()


def _find_embedding(subject: str, emb_dir: Path) -> Path | None:
    """First ``<subject>_*_emb.npy`` under ``emb_dir`` (one per subject here)."""
    hits = sorted(emb_dir.glob(f"{subject}_*_emb.npy"))
    return hits[0] if hits else None


def _clean(value):
    """Drop NaNs; pandas reads sparse UDS columns as floats."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def load_demo_cases(
    csv_path: str | Path | None = None,
    emb_dir: str | Path | None = None,
) -> list[DemoCase]:
    """Load every case in the demo cohort as a :class:`DemoCase`."""
    data_dir = _data_dir()
    csv_path = Path(csv_path) if csv_path else data_dir / "crychic_oasis12.csv"
    emb_dir = Path(emb_dir) if emb_dir else data_dir / "mri_emb"

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

        cases.append(DemoCase(
            id=rid,
            subject=subject,
            clinical=clinical,
            embedding_path=_find_embedding(subject, emb_dir),
            labels=labels,
        ))
    return cases
