"""Shared LLM transport for the agent steps (S1 extract, S3 router, S6 reasoner).

One async :func:`chat` entry point with two interchangeable providers, selected by
environment so the agent code never changes:

* **Claude (Anthropic API)** — the native ``anthropic`` SDK. Used when an Anthropic
  key is present (``ANTHROPIC_API_KEY`` / ``CLAUDE_API_KEY``) or
  ``CRYCHIC_LLM_PROVIDER=claude``. The agent system prompts are fixed and reused
  across the critic↔reviser loop and across cases, so each request marks the system
  block with ``cache_control`` — an identical prefix is then served from Anthropic's
  prompt cache. Opus 4.8 takes no sampling parameters, so the Claude path ignores
  ``temperature`` and uses the strict per-step system prompts to shape output.
* **Nebius / OpenAI-compatible** — the original ``NEBIUS_URL`` chat-completions
  transport, used when an endpoint is configured and Claude is not.

If neither is configured (or a request fails), :func:`chat` raises and each caller
degrades to its deterministic template — so the whole pipeline, the self-check loop
included, still runs offline.

This module holds transport ONLY. Prompts and fallbacks live with each agent step
(``agent/extract.py``, ``agent/router.py``, ``agent/reasoner.py``).
"""

from __future__ import annotations

import os

_DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"


def _provider() -> str | None:
    """Which backend :func:`chat` will use: ``'claude'`` | ``'nebius'`` | ``None``."""
    forced = os.environ.get("CRYCHIC_LLM_PROVIDER", "").strip().lower()
    if forced in ("claude", "anthropic"):
        return "claude"
    if forced in ("nebius", "openai"):
        return "nebius" if endpoint() else None
    if forced in ("offline", "none", "off"):
        return None
    # Auto: prefer Claude when a key is present, else the Nebius endpoint.
    if anthropic_api_key():
        return "claude"
    return "nebius" if endpoint() else None


# --- Claude (Anthropic) config ------------------------------------------------

def anthropic_api_key() -> str | None:
    """Anthropic key (the SDK also reads ``ANTHROPIC_API_KEY`` on its own)."""
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY") or None


def claude_model() -> str:
    return os.environ.get("CLAUDE_MODEL", _DEFAULT_CLAUDE_MODEL)


# --- Nebius / OpenAI-compatible config --------------------------------------

def endpoint() -> str | None:
    return os.environ.get("NEBIUS_URL") or None


def model() -> str:
    return os.environ.get("NEBIUS_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


def api_key() -> str | None:
    """Bearer token for the OpenAI-compatible endpoint; optional for self-hosted NIMs."""
    return os.environ.get("NEBIUS_API_KEY") or os.environ.get("NGC_API_KEY") or None


# --- shared -------------------------------------------------------------------

def online() -> bool:
    """True when some LLM backend is configured (does not guarantee reachability)."""
    return _provider() is not None


def provider_label() -> str:
    """Human-readable description of the active backend (for preflight logging)."""
    p = _provider()
    if p == "claude":
        return f"claude · {claude_model()}"
    if p == "nebius":
        return f"nebius · {model()}"
    return "offline"


async def chat(system: str, user: str, *, max_tokens: int = 1400,
               temperature: float = 0.2) -> str:
    """One chat completion via the configured backend. Raises if offline / on error.

    ``temperature`` is honoured only by the OpenAI-compatible path; the Claude path
    ignores it (Opus 4.8 does not accept sampling parameters).
    """
    p = _provider()
    if p == "claude":
        return await _chat_claude(system, user, max_tokens=max_tokens)
    if p == "nebius":
        return await _chat_openai(system, user, max_tokens=max_tokens,
                                  temperature=temperature)
    raise RuntimeError(
        "no LLM backend configured (set ANTHROPIC_API_KEY for Claude, or NEBIUS_URL)")


# --- Claude transport ---------------------------------------------------------

_claude_client = None  # cached AsyncAnthropic singleton (connection pooling)


def _get_claude_client():
    global _claude_client
    if _claude_client is None:
        from anthropic import AsyncAnthropic

        _claude_client = AsyncAnthropic()  # resolves ANTHROPIC_API_KEY from the env
    return _claude_client


async def _chat_claude(system: str, user: str, *, max_tokens: int) -> str:
    """One Anthropic Messages call; returns the concatenated text blocks.

    The system prompt is sent as a cacheable block: the per-step system prompts are
    byte-identical across the critic↔reviser loop and across cases, so repeated calls
    read the prefix from the prompt cache instead of re-billing it (a no-op below the
    model's cache minimum, which is fine). No ``thinking`` field → Opus 4.8 answers
    directly; ``thinking`` text, if a future model emits any, is dropped because only
    ``text`` blocks are read.
    """
    client = _get_claude_client()
    message = await client.messages.create(
        model=claude_model(),
        max_tokens=max_tokens,
        system=[{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in message.content if b.type == "text").strip()


# --- OpenAI-compatible (Nebius) transport -----------------------------------

async def _chat_openai(system: str, user: str, *, max_tokens: int,
                       temperature: float) -> str:
    """One OpenAI-compatible chat completion against ``NEBIUS_URL``."""
    url = endpoint()
    if not url:
        raise RuntimeError("NEBIUS_URL not configured")
    import httpx

    headers = {}
    key = api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    timeout = float(os.environ.get("NEBIUS_TIMEOUT", "60"))
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
