"""
UDS -> ADRD model-input mapping (vendored, verified).

The Xue 2024 model does NOT accept raw UDS variable names. Each feature must be
renamed with a section-specific prefix and its categorical codes remapped to
contiguous 0-based indices, e.g.:

    NACCMMSE (Neuropsych Battery) -> bat_NACCMMSE   (numeric, pass-through)
    NACCREAS (Health History)     -> his_NACCREAS    {1:0, 2:1, 7:2, 9:None}

Without this step the Formatter drops every feature, the imputer fills defaults,
and ``predict_proba`` returns an identical prior for every patient (silent
failure). This module reconstructs the mapping from the checkpoint's
``src_modalities`` + ``input_meta_info.csv`` — functionally verified equivalent
to the authors' ``nacc_variable_mappings.pkl``.

Extracted from ``smoke_test_xue2024_v3.py`` so the tool is self-contained.
"""
from __future__ import annotations

import json

import pandas as pd
import torch

# --- PyTorch 2.6+ load patch (checkpoint stores non-tensor metadata) --------
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# The 13 model outputs: cognitive stage (NC/MCI/DE) + 10 etiologies.
LABEL_COLS = ["NC", "MCI", "DE", "AD", "LBD", "VD", "PRD",
              "FTD", "NPH", "SEF", "PSY", "TBI", "ODE"]

# UDS form section -> model feature-name prefix (from checkpoint src_modalities).
SECTION_TO_PREFIX = {
    "Demographics": "his_",
    "Health History": "his_",
    "Family History": "his_",
    "Neuropsychological Battery Summary Scores": "bat_",
    "Physical/Neurological Exam Findings": "exam_",
    "Unified Parkinson's Disease Rating Scale (UPDRS)": "updrs_",
    "Medications": "med_",
    "Geriatric Depression Scale (GDS)": "gds_",
    "Hachinski Ischemic Score & Cerebrovascular Disease": "cvd_",
    "Neuropsychiatric Inventory Questionnaire (NPI-Q)": "npiq_",
    "Physical": "ph_",
    "Functional Activities Questionnaire (FAQ)": "faq_",
    "Genetic Data": "apoe_",
}


def build_mapping(src_modalities: dict, meta_df: pd.DataFrame) -> dict:
    """Reconstruct UDS-name -> (model_key, value_transform) from meta + checkpoint."""
    mapping = {}
    for _, row in meta_df.iterrows():
        name = row["Name"]
        prefix = SECTION_TO_PREFIX.get(row["Section"])
        if not prefix:
            continue
        model_key = f"{prefix}{name}"
        if model_key not in src_modalities:
            continue

        values_str = row["Values"].replace("'", '"').replace('"0": nan, ', "")
        try:
            values = json.loads(values_str)
        except (json.JSONDecodeError, ValueError):
            continue

        valid_codes = []
        for code_str, label in values.items():
            if code_str == "range":
                valid_codes = None  # numerical -> no transform
                break
            if label == "Unknown" or code_str in ("9", "99", "999"):
                continue
            valid_codes.append(int(code_str))

        transform_map = {} if valid_codes is None else {c: i for i, c in enumerate(valid_codes)}
        mapping[name] = (model_key, transform_map)
    return mapping


def convert_dictionary(original_dict: dict, mapping: dict, src_modalities: dict) -> dict:
    """Convert a raw UDS dict -> model-internal dict with strict validation.

    Categorical values whose remapped index falls outside ``num_categories`` are
    dropped; numerical values are coerced to float.
    """
    out = {}
    for k, v in original_dict.items():
        if k not in mapping:
            continue
        new_key, transform_map = mapping[k]
        info = src_modalities.get(new_key)
        if not info:
            continue
        try:
            v_int = int(v) if float(v) == int(float(v)) else float(v)
        except (ValueError, TypeError):
            continue
        if info["type"] == "categorical":
            if transform_map and v_int in transform_map:
                mapped = transform_map[v_int]
                if mapped < info.get("num_categories", 999):
                    out[new_key] = mapped
        else:
            try:
                out[new_key] = float(v)
            except (ValueError, TypeError):
                continue
    return out
