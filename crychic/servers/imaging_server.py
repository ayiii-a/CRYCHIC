#!/usr/bin/env python
"""CRYCHIC MCP server — structural imaging compute (S4).

Four deterministic-compute tools over the T1 (± FLAIR) imaging axis. The MONAI
bundles are loaded once at startup and stay resident (Inv #3); a tool call is
inference/geometry only. ``segment_t1`` runs at most once per case and every T1
metric is derived from that single cached segmentation (Inv #6). Numbers are
computed by ``geometry``/model code and travel with their threshold + reference —
never authored in prose (Inv #2). Binds localhost (Inv #10).

Tool descriptions are written as clinical actions because the router reads them as
dispatch signal (§7).

Run (CLAUDE.md §6):
    python crychic/servers/imaging_server.py        # streamable-http on :9901
    nat mcp client ping --url http://localhost:9901/mcp
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from crychic import overlay, segmentation, wmh  # noqa: E402
from crychic.schemas import ImagingCheck  # noqa: E402

mcp = FastMCP("crychic-imaging")


@mcp.tool()
def segment_t1(t1_path: str) -> dict[str, Any]:
    """Segment a T1 MRI into the 133-structure whole-brain parcellation (once per case).

    Runs MONAI wholeBrainSeg and caches the label map, so the hippocampal and
    ventricular metrics are both derived from this single segmentation without
    re-running inference. Returns segmented-structure counts, total intracranial
    (brain-tissue) volume, and key regional volumes.
    """
    return segmentation.segment(t1_path)


@mcp.tool()
def derive_metric(
    t1_path: str, check: str, age: float | None = None, sex: str | None = None,
) -> dict[str, Any]:
    """Quantify one structural biomarker from the cached T1 segmentation.

    ``check`` selects the biomarker, each with its own threshold and reference:
    ``hippo_z`` (hippocampal atrophy z-score for Alzheimer's disease, abnormal
    < -1.5) or ``evans`` (automated Evans-like screening flag for normal-pressure
    hydrocephalus, abnormal > 0.30). ``age``/``sex`` refine the hippocampal
    normative comparison. Returns the value, threshold, whether it is abnormal, and
    any approximation caveats — or an ``unavailable`` status (no fabricated value)
    if the structure could not be measured.
    """
    return segmentation.derive_metric(
        t1_path, ImagingCheck(check), age=age, sex=sex).model_dump(exclude_none=True)


@mcp.tool()
def wmh_fazekas(flair_path: str | None = None) -> dict[str, Any]:
    """Quantify white-matter-hyperintensity burden on FLAIR as an approximate Fazekas grade.

    Speaks to a vascular contribution (grade ≥ 2 flags vascular burden). Requires
    a FLAIR volume; when none is supplied this returns an ``unavailable`` status
    with an explicit "vascular burden not assessed from imaging" note rather than
    a fabricated grade — never implying VD was confirmed or cleared.
    """
    return wmh.fazekas(flair_path).model_dump(exclude_none=True)


@mcp.tool()
def render_overlay(t1_path: str, check: str, out_dir: str | None = None) -> dict[str, Any]:
    """Render the annotated key MRI slice for a structural finding.

    Produces a PNG of the slice that best shows the analyzed region, highlighted,
    with the finding's value / threshold / reference burned into the caption — the
    image a clinician verifies the number against. Returns the PNG path (or null
    if the T1 has not been segmented).
    """
    rendered = overlay.render_overlay(t1_path, ImagingCheck(check), out_dir=out_dir)
    return {"t1_path": t1_path, "check": check,
            "png_path": rendered["png_path"] if rendered else None,
            "plane": rendered["plane"] if rendered else None,
            "index": rendered["index"] if rendered else None}


def main() -> None:
    mcp.settings.host = os.environ.get("MCP_HOST", "127.0.0.1")  # Inv #10
    mcp.settings.port = int(os.environ.get("IMAGING_PORT", "9901"))
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
    if transport != "stdio":
        for name, fn in (("wholeBrainSeg", segmentation.warmup), ("WMH", wmh.warmup)):
            try:  # load bundles once at startup (Inv #3)
                fn()
            except Exception as exc:
                print(f"[imaging] {name} warmup deferred (loads on first call): {exc}",
                      file=sys.stderr)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
