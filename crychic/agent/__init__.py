"""CRYCHIC agent logic — the reasoning steps, which are NOT MCP tools (CLAUDE.md §2.4).

    extract.py    S1  clinical note → Xue feature dict (+ confidences)
    router.py     S3  the one real LLM decision — which imaging checks to dispatch
    reasoner.py   S4d guardrailed finding sentences + S6 reconciliation report
    workflow.yml      NAT mcp_client config that wires these over the MCP servers

Each step calls the configured LLM through :mod:`crychic.llm_client` and falls
back to a deterministic template offline, so the pipeline runs with or without a
live endpoint. The flow is a fixed spine with exactly one LLM *decision* (the
router, S3) — not a free-roaming ReAct loop (Inv #5).
"""
