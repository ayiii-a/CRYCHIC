"""CRYCHIC — multi-modal dementia clinical decision support.

A multimodal dementia differential (Xue 2024, clinical features + an optional
SwinUNETR MRI embedding) cross-checked against quantitative structural imaging
biomarkers (MONAI), presented as radiology-style finding cards a clinician reviews
and signs. Deterministic compute is exposed as MCP tools (``crychic/servers``); the
router and reasoner are agent logic (``crychic/agent``); the spine that wires them
is ``crychic/pipeline.py``. See ``CLAUDE.md`` for the architecture and invariants.
"""

__version__ = "0.4.0"
