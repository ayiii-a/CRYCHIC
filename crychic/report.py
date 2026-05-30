"""S7 — self-contained clinical report (printable HTML) with embedded key slices.

Renders a :class:`~crychic.schemas.UnifiedEvidence` bundle into one portable HTML
document: the CDS disclaimers, the clinical screening summary, each finding as an
impression + its annotated key slice (base64-embedded, so the file stands alone and
prints to PDF), the reconciliation, conflicts, options, limitations, references, and
the sign-off bar.

The numbers are still the metric's own fixed tokens (Inv #2); the embedded PNG is
only the verifiable key slice the clinician checks the number against. This is a
*rendering* of already-compliant content — it does not author anything new.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

from . import cds_guard
from .agent import reasoner
from .schemas import FindingCard, ReconClass, UnifiedEvidence

_POLARITY = {
    "supporting": ("#b45309", "#fffbeb", "supporting"),
    "negative": ("#047857", "#ecfdf5", "normal"),
    "abstain": ("#475569", "#f1f5f9", "not assessed"),
}
_RECON = {
    ReconClass.CONCORDANT: ("#047857", "concordant — imaging supports the signal"),
    ReconClass.DISCORDANT: ("#b91c1c", "discordant — imaging does not support it"),
    ReconClass.CLINICAL_ONLY: ("#0369a1", "clinical-only — no imaging axis"),
    ReconClass.INCIDENTAL: ("#b45309", "incidental — imaging without a strong signal"),
}
_SEVERITY = {"info": "#475569", "caution": "#b45309", "important": "#b91c1c"}

_CSS = """
:root { --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --accent:#0369a1; }
* { box-sizing: border-box; }
body { font: 14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       color: var(--ink); background:#f8fafc; margin:0; }
.page { max-width: 820px; margin: 0 auto; background:#fff; padding: 28px 34px 40px;
        box-shadow: 0 1px 4px rgba(0,0,0,.08); }
.hdr { display:flex; align-items:flex-start; gap:12px; border-bottom:2px solid var(--accent);
       padding-bottom:10px; margin-bottom:6px; }
.hdr .t { font-size:20px; font-weight:700; }
.hdr .s { font-size:11px; color:var(--muted); }
.disc { background:#eff6ff; border:1px solid #bfdbfe; color:#1e3a8a; border-radius:6px;
        padding:8px 11px; font-size:12px; margin:10px 0 16px; }
h2 { font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--accent);
     border-bottom:1px solid var(--line); padding-bottom:3px; margin:20px 0 8px; }
.meta { font-size:13px; color:#334155; }
.meta b { color:var(--ink); }
.finding { display:flex; gap:14px; border:1px solid var(--line); border-radius:8px;
           padding:11px 13px; margin:9px 0; page-break-inside:avoid; }
.finding .body { flex:1; }
.finding img { width:210px; height:auto; border-radius:6px; background:#0b1220; }
.chip { display:inline-block; font-size:10px; font-weight:700; padding:1px 7px;
        border-radius:10px; vertical-align:middle; }
.title { font-weight:600; }
.sent { margin:5px 0; color:#334155; }
.metric { font-size:12px; color:#0f172a; } .metric b { font-variant-numeric: tabular-nums; }
.ref { font-size:10.5px; color:var(--muted); margin-top:3px; }
table { width:100%; border-collapse:collapse; font-size:12.5px; }
th,td { text-align:left; padding:5px 7px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font-weight:600; }
ul { margin:6px 0 6px 18px; padding:0; } li { margin:3px 0; color:#334155; }
.opt { list-style:none; margin-left:2px; } .opt li::before { content:"○ "; color:var(--accent); }
.foot { border-top:2px solid var(--accent); margin-top:22px; padding-top:10px;
        font-size:12px; color:var(--muted); }
.signoff { margin-top:8px; }
.btn { display:inline-block; border-radius:6px; padding:5px 11px; font-weight:600; font-size:12px;
       margin-right:6px; border:1px solid; }
.b-agree { color:#047857; border-color:#34d399; background:#ecfdf5; }
.b-edit  { color:#334155; border-color:#cbd5e1; background:#f1f5f9; }
.b-dis   { color:#b91c1c; border-color:#fca5a5; background:#fef2f2; }
@media print { body { background:#fff; } .page { box-shadow:none; max-width:none; }
               @page { size:A4; margin:16mm; } }
"""


def _esc(s: object) -> str:
    return html.escape(str(s))


def _img_data_uri(path: str | None) -> str | None:
    """Base64 data URI for a PNG path, so the report is self-contained. None if absent."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _finding_html(c: FindingCard) -> str:
    color, bg, label = _POLARITY.get(c.polarity, _POLARITY["abstain"])
    chip = (f'<span class="chip" style="color:{color};background:{bg};'
            f'border:1px solid {color}33">{_esc(label)}</span>')
    metric_line = ""
    if c.metric and c.metric.status.value == "measured":
        m = c.metric
        unit = f" {_esc(m.unit)}" if m.unit else ""
        metric_line = (f'<div class="metric"><b>{_esc(m.value)}{unit}</b> '
                       f'(threshold {_esc(m.comparator)} {_esc(m.threshold)})</div>')
    refs = (f'<div class="ref">{_esc("; ".join(c.references))}</div>'
            if c.references else "")
    uri = _img_data_uri(c.overlay_png_path)
    img = (f'<img src="{uri}" alt="{_esc(c.title)} key slice">' if uri else "")
    slice_note = (f' &middot; key slice: {_esc(c.key_slice.plane)} #{c.key_slice.index}'
                  if c.key_slice else "")
    return (
        f'<div class="finding"><div class="body">'
        f'<div class="title">{_esc(c.title)} {chip}</div>'
        f'<div class="sent">{_esc(c.sentence)}{slice_note}</div>'
        f'{metric_line}{refs}</div>{img}</div>'
    )


def render_report_html(u: UnifiedEvidence) -> str:
    """Render the full clinical report as one self-contained HTML string."""
    d = u.differential
    if d.imaging_used:
        screen_heading = "Dementia screening (Xue 2024 — clinical + MRI, multimodal)"
        independence_note = ("Probabilities are non-exclusive and consistent with, not "
                             "confirmatory of, any single etiology; the MRI embedding "
                             "informs them, so the structural findings below are a "
                             "consistency cross-check on imaging-informed probabilities, "
                             "not a fully independent axis.")
    else:
        screen_heading = "Clinical screening (Xue 2024 — clinical features only)"
        independence_note = ("Probabilities are non-exclusive and consistent with, not "
                             "confirmatory of, any single etiology; imaging is assessed "
                             "independently (no MRI is fed to the clinical model).")

    findings = "".join(_finding_html(c) for c in u.cards) or \
        '<div class="sent">No imaging findings for this case.</div>'

    recon_rows = "".join(
        f'<tr><td><b>{_esc(r.etiology)}</b></td><td>{r.prob:.2f}</td>'
        f'<td style="color:{_RECON[r.recon][0]}">{_esc(_RECON[r.recon][1])}</td>'
        f'<td>{_esc(" ".join(r.evidence))}</td></tr>'
        for r in u.reconciliations) or '<tr><td colspan="4">No etiology crossed the thresholds.</td></tr>'

    if u.conflicts:
        conflicts = "<ul>" + "".join(
            f'<li><b style="color:{_SEVERITY.get(c.severity.value, "#475569")}">'
            f'{_esc(c.name)}</b> [{_esc(c.severity.value)}]: {_esc(c.description)}</li>'
            for c in u.conflicts) + "</ul>"
    else:
        conflicts = '<p class="sent">No internal evidence conflicts were detected, which ' \
                    'does not exclude the alternatives below.</p>'

    limitations = "<ul>" + "".join(
        f"<li>{_esc(line.lstrip('- '))}</li>" for line in reasoner._limitations(u)) + "</ul>"
    references = "<ul>" + "".join(f"<li>{_esc(r)}</li>" for r in cds_guard.REFERENCES) + "</ul>"

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CRYCHIC Clinical Evidence Report — {_esc(u.case_id)}</title>
<style>{_CSS}</style></head><body><div class="page">
  <div class="hdr"><div style="font-size:26px">⚕️</div><div>
    <div class="t">CRYCHIC — Clinical Evidence Report</div>
    <div class="s">Decision support, not a diagnosis &middot; case <b>{_esc(u.case_id)}</b></div>
  </div></div>

  <div class="disc"><b>Clinical Decision Support — not a diagnosis.</b> CRYCHIC organizes
  multi-modal evidence to assist a qualified clinician. It does not establish a diagnosis,
  prescribe, or direct care. The clinician makes and is responsible for the final decision.
  All values are model-derived and must be verified against the source images and the full
  clinical context.</div>

  <h2>{_esc(screen_heading)}</h2>
  <div class="meta">Most likely cognitive stage: <b>{_esc(d.stage_top)}</b>
    (P(MCI)={d.p_mci:.2f}, P(DE)={d.p('DE'):.2f}, P(NC)={d.p('NC'):.2f}).<br>
    Leading etiology signal: <b>{_esc(d.etiology_top)}</b>
    (P(AD)={d.p_ad:.2f}, P(VD)={d.p_vd:.2f}, P(FTD)={d.p('FTD'):.2f}, P(NPH)={d.p('NPH'):.2f}).<br>
    <span class="ref">{_esc(independence_note)}</span>
  </div>

  <h2>Imaging findings</h2>
  {findings}

  <h2>Reconciliation — clinical signal vs imaging</h2>
  <table><tr><th>Etiology</th><th>P</th><th>Concordance</th><th>Evidence</th></tr>
  {recon_rows}</table>

  <h2>Differential considerations &amp; counter-evidence</h2>
  {conflicts}
  <p class="sent" style="margin-top:6px">Alternatives the clinician may weigh:</p>
  <ul class="opt">
    <li>A non-AD neurodegenerative process (e.g. FTD-spectrum, LATE) if the course or atrophy
        emphasis diverges from the hippocampal picture.</li>
    <li>A vascular or mixed contribution, particularly if the clinical vascular signal is
        elevated (FLAIR would clarify).</li>
    <li>A reversible / functional contributor (mood, metabolic, medication) pending follow-up.</li>
  </ul>

  <h2>Options for the clinician to consider</h2>
  <ul class="opt">
    <li>Correlate this evidence with the history, examination, and prior imaging before any decision.</li>
    <li>Consider confirmatory work-up where an axis is uncertain or unavailable.</li>
    <li>Consider longitudinal follow-up to clarify trajectory.</li>
  </ul>

  <h2>Limitations</h2>
  {limitations}

  <h2>References</h2>
  {references}

  <div class="foot">
    This draft is <b>not part of the medical record</b> until a clinician reviews and signs it.
    <div class="signoff">
      <span class="btn b-agree">✓ Agree &amp; sign</span>
      <span class="btn b-edit">✎ Edit</span>
      <span class="btn b-dis">✗ Disagree</span>
    </div>
  </div>
</div></body></html>"""
