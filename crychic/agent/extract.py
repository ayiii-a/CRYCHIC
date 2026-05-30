"""S1 — feature extraction: clinical note → Xue feature dict (+ confidences).

Two paths, both returning a typed :class:`~crychic.schemas.XueFeatures`:

* **Structured (the demo path).** A UDS dict or a ``.json``/``.csv`` record is
  passed through verbatim, with confidence 1.0 — no LLM involved.
* **Free text.** A clinical note is parsed by the LLM into a small set of raw UDS
  variables with per-field confidence. Offline (or on any failure) this degrades
  to an empty feature set with a caveat, and downstream Xue returns its prior.

This step only produces the *clinical* model inputs; it never authors a number that
appears as a *finding* in the report (those come from compute, Inv #2). The MRI
embedding fed alongside these features to the multimodal xue_predict is resolved
separately by the caller (``cases.find_emb``), not here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .. import llm_client
from ..schemas import XueFeatures

# The compact UDS target set the extractor aims for (others pass through if given).
_TARGET_VARS = {
    "NACCAGE": "age in years",
    "SEX": "1 = male, 2 = female (NACC coding)",
    "EDUC": "years of education",
    "NACCMMSE": "MMSE total, 0–30",
    "NACCNE4S": "number of APOE ε4 alleles, 0/1/2",
    "CDRGLOB": "global CDR, 0/0.5/1/2/3",
}

_EXTRACT_SYSTEM = (
    "You extract structured UDS variables from a free-text clinical note for a "
    "dementia screening model. Return ONLY a compact JSON object mapping UDS "
    "variable names to numeric values for the fields you can infer with reasonable "
    "confidence; omit anything not stated. Use NACC coding (SEX: 1=male, 2=female). "
    "Do not guess, do not add prose. Target variables:\n"
    + "\n".join(f"  {k}: {d}" for k, d in _TARGET_VARS.items())
)

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def _load_record(path: Path) -> dict:
    """Load a single clinical record from .json or .csv (first row)."""
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() == ".csv":
        import pandas as pd
        return pd.read_csv(path).iloc[0].dropna().to_dict()
    raise ValueError(f"Unsupported clinical record: {path!r}")


def _structured(features: dict) -> XueFeatures:
    return XueFeatures(
        features=dict(features),
        confidences={k: 1.0 for k in features},
        source="structured",
    )


async def extract_features_async(clinical: dict | str | Path) -> XueFeatures:
    """Resolve ``clinical`` into :class:`XueFeatures` (S1).

    A dict or a path to a record is structured pass-through; free prose is parsed
    by the LLM (empty + caveat on failure / offline).
    """
    if isinstance(clinical, dict):
        return _structured(clinical)

    if isinstance(clinical, (str, Path)):
        p = Path(clinical)
        if p.exists() and p.suffix.lower() in (".json", ".csv"):
            try:
                return _structured(_load_record(p))
            except Exception as exc:
                return XueFeatures(source="structured",
                                   caveats=[f"Could not load clinical record {p}: {exc}"])
        # Otherwise treat the string as a free-text clinical note → LLM extraction.
        return await _extract_from_note(str(clinical))

    return XueFeatures(source="structured",
                       caveats=["No usable clinical input; model returns its prior."])


async def _extract_from_note(note: str) -> XueFeatures:
    try:
        raw = await llm_client.chat(_EXTRACT_SYSTEM, note, max_tokens=400, temperature=0.0)
        m = _JSON_OBJ.search(raw)
        parsed = json.loads(m.group(0)) if m else {}
        features = {k: v for k, v in parsed.items() if isinstance(v, (int, float))}
    except Exception as exc:
        return XueFeatures(
            source="extracted",
            caveats=[f"Free-text feature extraction unavailable ({exc}); "
                     "the model will return its prior. Provide structured UDS "
                     "variables for a real screen."])
    return XueFeatures(
        features=features,
        confidences={k: 0.7 for k in features},
        source="extracted",
        caveats=["Features were extracted from free text by an LLM (confidence ≈ "
                 "0.7); verify against the source note before relying on the screen."]
        if features else
        ["No UDS variables could be extracted from the note; model returns its prior."],
    )
