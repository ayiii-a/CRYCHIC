"""Deterministic Clinical-Decision-Support compliance checker (CLAUDE.md §2.9).

The single source of truth for whether a report obeys the CDS output rules. It is
pure string analysis — no LLM — so the self-check loop can never be talked past:
even when an online critic says "PASS", a report that is actually missing its
sign-off footer (or uses a forbidden directive verb) still fails here.

The six principles, and what is checked:
    1  Hedged language        no "recommend", "the diagnosis is", "prescribe", ...
    2  Header + footer        both CDS disclaimers present
    3  Traceable claims       a References section + threshold symbols (≥/≤/</>) present
    4  Counter-evidence       a non-empty differential AND a non-empty limitations section
    5  Options not orders     '○' option bullets, never a numbered recommendation list
    6  Sign-off               the [Agree & sign] [Edit] [Disagree] buttons in the footer

Also holds the fixed CDS boilerplate (header/footer/sign-off/references) that every
report carries verbatim, so the language model is never in a position to paraphrase
a disclaimer away.
"""

from __future__ import annotations

import re

_HEADER_MARK = "Clinical Decision Support — not a diagnosis"
_FOOTER_MARK = "Decision support only"
SIGN_OFF = "[✓ Agree & sign] [✏️ Edit] [✗ Disagree]"

CDS_HEADER = (
    f"> ⚕️ **{_HEADER_MARK}.** CRYCHIC organizes multi-modal evidence to assist a "
    "qualified clinician. It does not establish a diagnosis, prescribe, or direct "
    "care. The clinician makes and is responsible for the final decision."
)
CDS_FOOTER = (
    "---\n"
    f"> ⚕️ **{_FOOTER_MARK}.** This draft is not part of the medical record until a "
    "clinician reviews and signs it. All values are model-derived and must be "
    "verified against the source images and the full clinical context.\n\n"
    f"{SIGN_OFF}"
)

# Structural-axis references (amyloid/PET references intentionally removed — §8).
REFERENCES = [
    "Xue et al., *Nature Medicine* 2024 — AI differential diagnosis of dementia (clinical model).",
    "NIA-AA 2018 — A/T/(N) biological framework; structural neurodegeneration criteria.",
    "Evans 1942 — ventricular (Evans) index; here approximated as an Evans-like index.",
    "Fazekas et al. 1987 / Wardlaw STRIVE — white-matter-hyperintensity grading & SVD standards.",
    "MONAI wholeBrainSeg (UNEST, 133-label whole-brain parcellation).",
]

# Directive / definitive phrasings that violate principle #1 (hedged language).
# Judged on the BODY only, so the disclaimers' own "does not establish a diagnosis"
# wording is never flagged.
_FORBIDDEN = [
    r"\brecommend(s|ed|ation|ations)?\b",
    r"\bthe diagnosis is\b",
    r"\bis diagnosed (with|as)\b",
    r"\bwe advise\b",
    r"\bprescrib(e|es|ed|ing)\b",
    r"\bstart (the )?patient on\b",
    r"\bmust (start|be started|prescribe)\b",
]


def _strip_boilerplate(markdown: str) -> str:
    """Body with the fixed header/footer disclaimers removed (for principle #1)."""
    body = markdown
    for chunk in (CDS_HEADER, CDS_FOOTER):
        body = body.replace(chunk, "")
    return body


def _section_nonempty(markdown: str, keyword: str) -> bool:
    """True if a heading containing ``keyword`` exists with body text under it."""
    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#") and keyword.lower() in line.lower():
            for nxt in lines[i + 1:]:
                if nxt.lstrip().startswith("#"):
                    break
                if nxt.strip():
                    return True
    return False


def cds_violations(markdown: str) -> list[str]:
    """Check all 6 CDS principles. Empty list == compliant (PASS)."""
    v: list[str] = []
    body = _strip_boilerplate(markdown)
    low = markdown.lower()

    # 1 — hedged language.
    for pat in _FORBIDDEN:
        m = re.search(pat, body, flags=re.IGNORECASE)
        if m:
            v.append(f"Principle 1 (hedged language): forbidden phrasing {m.group(0)!r}.")
            break

    # 2 — header + footer disclaimers present.
    if _HEADER_MARK.lower() not in low:
        v.append("Principle 2: CDS header disclaimer missing.")
    if _FOOTER_MARK.lower() not in low:
        v.append("Principle 2: CDS footer disclaimer missing.")

    # 3 — traceable claims: references section + threshold symbols present.
    if "references" not in low:
        v.append("Principle 3: no References section for traceable claims.")
    if not re.search(r"[≥≤<>]", markdown):
        v.append("Principle 3: numeric claims lack thresholds (no ≥/≤/< present).")

    # 4 — counter-evidence: differential + limitations sections, non-empty.
    if not _section_nonempty(markdown, "differential"):
        v.append("Principle 4: differential / counter-evidence section empty or missing.")
    if not _section_nonempty(markdown, "limitation"):
        v.append("Principle 4: limitations section empty or missing.")

    # 5 — options not recommendations: '○' bullets, no numbered lists.
    if "○" not in markdown:
        v.append("Principle 5: options must use '○' bullets (none found).")
    if re.search(r"(?m)^\s*\d+\.\s", body):
        v.append("Principle 5: numbered list present — options must not be numbered.")

    # 6 — sign-off buttons in footer.
    if SIGN_OFF not in markdown:
        v.append("Principle 6: sign-off buttons missing from footer.")

    return v


def ensure_boilerplate(markdown: str) -> str:
    """Make sure header/footer survive an LLM round-trip (cheap guardrail)."""
    out = markdown
    if _HEADER_MARK not in out:
        out = CDS_HEADER + "\n\n" + out
    if SIGN_OFF not in out:
        out = out.rstrip() + "\n\n" + CDS_FOOTER
    return out
