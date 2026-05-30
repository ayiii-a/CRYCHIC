#!/usr/bin/env python
"""CRYCHIC MCP server — the clinical differential (S2).

Exposes a single deterministic-compute tool, ``xue_predict``, over the Xue 2024
(Nature Medicine) ADRD model. Clinical features ONLY — no imaging ever reaches
this server (Inv #1). The model is loaded once at startup and stays resident
(Inv #3); a tool call is inference only. Binds localhost (Inv #10).

Run (CLAUDE.md §6):
    python crychic/servers/xue_server.py            # streamable-http on :9902
    MCP_TRANSPORT=stdio python crychic/servers/xue_server.py   # for the MCP Inspector
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

# Allow direct execution (`python crychic/servers/xue_server.py`) by putting the
# repo root on sys.path so `import crychic` resolves.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from crychic import xue  # noqa: E402

mcp = FastMCP("crychic-xue")


@mcp.tool()
def xue_predict(clinical: dict[str, Any], explain: bool = False) -> dict[str, Any]:
    """Estimate a 13-label dementia differential from structured UDS clinical variables.

    Clinical features ONLY — this tool never accepts or uses imaging, so the
    differential stays independent of the structural axis. Pass ``clinical`` as
    raw UDS variables (e.g. ``{"NACCAGE": 78, "NACCMMSE": 22, "SEX": 2,
    "NACCNE4S": 1}``). Returns the cognitive-stage probabilities (NC/MCI/DE), the
    10 etiology probabilities (AD, LBD, VD, FTD, NPH, ...), and — when
    ``explain`` is set — a clinical feature × label attribution heatmap.
    """
    return xue.screen(clinical, explain=explain).model_dump(exclude_none=True)


def main() -> None:
    mcp.settings.host = os.environ.get("MCP_HOST", "127.0.0.1")  # Inv #10
    mcp.settings.port = int(os.environ.get("XUE_PORT", "9902"))
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
    if transport != "stdio":
        try:  # load weights once at startup (Inv #3)
            xue.warmup()
        except Exception as exc:  # don't block serving if weights are absent in dev
            print(f"[xue] model warmup deferred (loads on first call): {exc}",
                  file=sys.stderr)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
