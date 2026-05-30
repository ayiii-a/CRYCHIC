"""CRYCHIC MCP servers — the deterministic-compute surface (CLAUDE.md §2.4, §5).

Two warm, local-only MCP servers expose pure compute as clean tools:

    xue_server      :9902  xue_predict                                    (the differential)
    imaging_server  :9901  segment_t1, derive_metric, wmh_fazekas, render_overlay

Reasoning is NOT exposed here — the router (S3) and reasoner (S6) are agent logic
(see ``crychic/agent/``), and the aggregator (S5) is plain code. Both servers use
the official ``mcp.server.fastmcp`` runtime (NOT ``nvidia-nat-fastmcp`` — §7) and
bind to localhost only (Inv #10).
"""
