# CRYCHIC

> **CRYCHIC**: multi-modal dementia clinical decision support.
> Submission for the Healthcare Agentic AI Hackathon (Harvard Innovation Labs, 2026-05-30).

---

## What This Is

CRYCHIC takes a **clinical note + brain MRI (T1, optionally FLAIR)** and produces a
**clinician-facing decision-support report**: a **multimodal** differential over 13
dementia etiologies (Xue 2024, *clinical features + a SwinUNETR MRI embedding*),
cross-checked against quantitative structural imaging biomarkers (MONAI) whose
numbers are computed **independently in geometry code**, presented as radiology-style
**finding cards** — number + threshold + reference + an annotated key image — that a
clinician reviews and signs.

It is **Clinical Decision Support, not a diagnostic device.** The clinician keeps the
decision and the responsibility. The framing is *"AI shows you the evidence — you
decide,"* never *"AI says X, do X."*

All inference is local; no PHI leaves the box (MCP servers bind to localhost).

> See [`CLAUDE.md`](CLAUDE.md) for the design invariants. The credibility anchor:
> **numbers are computed, never authored by an LLM** — a finding's digits are fixed
> tokens injected from `geometry.py`/tool code, so the model is structurally unable
> to put a wrong number in the report.

---

## Architecture at a Glance

```
clinical_note + T1 (+ optional FLAIR)
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Agent (NAT / in-process spine)  —  one LLM decision: the router (S3)        │
│   S1 extract → S2 xue_predict → S3 route → S4 imaging+translate             │
│             → S5 aggregate (plain code) → S6 reason+self-check → S7 render   │
└───────────────┬───────────────────────────────────────┬───────────────────┘
                │ MCP (localhost)                         │ OpenAI-compatible
                ▼                                         ▼
┌───────────────────────────────┐          ┌───────────────────────────────┐
│ MCP: imaging_server  :9901     │          │ Nebius NIM / Claude endpoint │
│   segment_t1                   │          │  (router rationale, reasoner,  │
│   derive_metric (geometry)     │          │   critic↔reviser self-check)   │
│   wmh_fazekas                  │          └───────────────────────────────┘
│   render_overlay               │
├───────────────────────────────┤   Deterministic compute = MCP tools.
│ MCP: xue_server      :9902     │   Reasoning (router, reasoner) = agent logic.
│   xue_predict (clinical+MRI)   │   Aggregation (S5) = plain code, not a tool.
└───────────────────────────────┘
```

**Key design choices** (CLAUDE.md §2):
- **Xue runs multimodal** (clinical features + the SwinUNETR MRI embedding). The
  structural finding-card *numbers* are still computed independently in `geometry.py`,
  so S6 reconciliation is a consistency cross-check on imaging-informed probabilities.
  (Omit the embedding to recover a fully independent second opinion — the pipeline
  falls back to clinical-only and the report wording adapts.)
- **The flow is a fixed spine with exactly one LLM decision** (the S3 router, which
  picks the extra modality-gated imaging check); everything else executes in order.
- **Models load once at server startup and stay resident**; a tool call is inference only.

---

## The Pipeline (the spine)

| Step | Role | Type |
|---|---|---|
| S1 extract | clinical note → Xue feature dict (+ confidences) | agent (LLM) |
| S2 `xue_predict` | 13-label differential, **clinical + MRI embedding (multimodal)** | MCP tool |
| S3 router | **decide which imaging checks to dispatch** | agent — the one decision |
| S4a `segment_t1` | 133-structure whole-brain seg, **once** per case | MCP tool |
| S4b `derive_metric` | hippocampal z-score / Evans-like screening flag (geometry) | MCP tool |
| S4c `wmh_fazekas` | WMH → Fazekas (FLAIR; abstains without it) | MCP tool |
| S4d translate | guardrailed finding sentence + key slice + overlay | agent + deterministic |
| S5 aggregate | merge + reconcile + conflicts + provenance | plain code |
| S6 reason | reconcile probs vs cards; CDS report + self-check | agent (LLM) |
| S7 render | cards as impression + image + caption; sign-off | UI |

Reconciliation (S6): high prob + supporting imaging → **concordant**; high +
contradicting → **discordant, flag**; high + no axis → **clinical-only**; low +
positive imaging → **incidental**.

---

## Etiology → imaging mapping

| Label | Metric | Threshold | Modality |
|---|---|---|---|
| AD | hippocampal z-score (age/sex/TIV-adjusted) | < -1.5 | T1 (free) |
| NPH | automated Evans-like index *(screening flag)* | > 0.30 | T1 (free) |
| VD | WMH volume → Fazekas | ≥ 2 | FLAIR (intensity proxy; optional WMH bundle) |
| FTD / PRD / SEF / PSY / TBI / LBD | — | — | **abstain** (no defended structural correlate) |
| ODE | tumour (BraTS) | — | niche — not built; abstain |

The two T1 metrics come from the *one* segmentation, so the free structural
baseline always runs when a T1 is present — and a **normal result still produces a
finding card** ("hippocampus normal, no structural support for AD"), because a
negative is as much CDS as a positive. The frontotemporal lobar-Z check was **retired
in v0.5** (its normative fraction was synthetic, not atlas-matched), so FTD now
abstains; the Evans-like index is kept but framed as a coarse **screening flag**, not
a diagnostic Evans measurement. Etiologies with no defended structural correlate are
**explicitly abstained** — CRYCHIC never implies imaging confirmed or cleared them.

---

## MCP Tools

`imaging_server.py` (`:9901`) — `segment_t1`, `derive_metric`, `wmh_fazekas`,
`render_overlay`.
`xue_server.py` (`:9902`) — `xue_predict`.

Both use the official `mcp.server.fastmcp` runtime (not `nvidia-nat-fastmcp`),
bind to localhost, and load their models once at startup. Reasoning is **not**
exposed as a tool — the router and reasoner are agent logic in `crychic/agent/`.

---

## The 6 CDS Principles (enforced by the self-check loop)

| # | Principle | Checked by `cds_guard` |
|---|---|---|
| 1 | Hedged language | no "diagnose"/"recommend"/imperatives |
| 2 | CDS header/footer | both disclaimers present |
| 3 | Traceable claims | References section + threshold symbols |
| 4 | Counter-evidence | non-empty differential & limitations |
| 5 | Options not orders | `○` bullets, never numbered |
| 6 | Sign-off requirement | footer states the draft requires clinician sign-off (the sign-off control lives in the app UI, not the report body) |

The deterministic checker is authoritative: an online critic can add findings but
can never wave through a report that actually violates a principle. Only the signed
version enters the record; the draft never auto-commits.

---

## Repository Layout

```
crychic/
├── schemas.py            # Pydantic contracts — shared by servers AND agent
├── checks.py             # single source of truth for the tier-2 structural checks
├── geometry.py           # deterministic math: hippo z-score, Evans-like, slice picker
├── overlay.py / render.py# annotated key-slice rendering
├── aggregate.py          # S5 plain code: reconcile + conflicts + provenance
├── report.py             # S7 self-contained printable HTML (embedded key slices)
├── cds_guard.py          # deterministic 6-principle CDS checker + boilerplate
├── llm_client.py         # shared Nebius/OpenAI transport + offline fallback
├── xue.py                # clinical-only Xue 2024 wrapper (no MRI ever)
├── segmentation.py       # MONAI wholeBrainSeg singleton + seg cache + derive_metric
├── wmh.py                # FLAIR WMH → Fazekas (abstains without FLAIR)
├── pipeline.py           # the in-process S1→S7 spine
├── state.py / cases.py   # ephemeral case store; demo-cohort loader
├── norms/hippo_wscore.json   # age/sex/TIV Z-score coefficients
├── servers/              # MCP: imaging_server.py, xue_server.py
└── agent/                # router.py, reasoner.py, extract.py, workflow.yml (NAT)
crychic_web.py            # FastAPI UI over the spine (localhost)
webui/index.html          # single-page finding-card UI
scripts/                  # run_one_case.py, inspect_cases.py, fit_hippo_wscore.py,
                          #   coreg_flair.py (FLAIR→T1 rigid coreg), smoke_offline.py
```

---

## Quick Start

```bash
pip install -r requirements.txt        # torch/monai/nibabel + mcp/pydantic/fastapi/...

# Run a case end-to-end via the in-process spine (no servers / NAT needed).
# Agent steps on Claude (native SDK) — just export the key:
ANTHROPIC_API_KEY=sk-ant-... \
TIER1_DEVICE=cuda TIER2_DEVICE=cuda \
  python scripts/run_one_case.py --case OAS30209 --report report.html

# …or point the same step at the OpenAI-compatible Nebius (Token Factory) endpoint:
NEBIUS_URL=https://api.studio.nebius.com/v1/chat/completions \
NEBIUS_API_KEY=... NEBIUS_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  python scripts/run_one_case.py --case OAS30209

# Or the web UI (localhost only):
uvicorn crychic_web:app --host 127.0.0.1 --port 8080

# Enable the VD/Fazekas axis: fetch a subject's FLAIR (OASIS-3, your XNAT creds),
# then rigidly coregister it onto the MNI T1 grid so the pipeline picks it up:
python scripts/coreg_flair.py --subject OAS30073 --flair /path/to/FLAIR.nii.gz
```

The multimodal Xue model auto-uses a subject's precomputed SwinUNETR embedding
(`data/mri_emb/<subject>_*_emb.npy`) when present; with none, it falls back to a
clinical-only differential and the report wording adapts.

The LLM steps are optional: with no backend configured (no `ANTHROPIC_API_KEY` and
no `NEBIUS_URL`), every agent step (extract, router rationale, reasoner, critic)
falls back to a deterministic template, so the
whole pipeline — self-check included — still runs offline.

### Running the MCP servers (CLAUDE.md §6)

```bash
python crychic/servers/imaging_server.py   # :9901, loads MONAI bundles once
python crychic/servers/xue_server.py       # :9902, loads the Xue model once

nat mcp client ping --url http://localhost:9901/mcp       # health check
# NAT workflow over the servers (verify node decorators vs your NAT version — §7):
nat run --config_file crychic/agent/workflow.yml --input cases/case_C.json
```

Demo cohort (T1-only, OASIS-3): the bundled `data/crychic_oasis12.csv` + per-subject
T1s under `data/mri_prepro/`. Representative cases — **A** ≈ CN/normal, **B** ≈
MCI/borderline, **C** ≈ typical AD.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` / `CLAUDE_API_KEY` | — | **Claude backend** — set this to run the agent steps on Claude (native SDK) |
| `CLAUDE_MODEL` | `claude-opus-4-8` | Claude model id for the agent steps |
| `CRYCHIC_LLM_PROVIDER` | (auto) | force a backend: `claude` \| `nebius` \| `offline` (auto: Claude if a key is set, else Nebius, else offline) |
| `NEBIUS_URL` | (unset) | OpenAI-compatible chat endpoint, e.g. `https://api.studio.nebius.com/v1/chat/completions` (the alternative backend) |
| `NEBIUS_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | model name in the request body |
| `NEBIUS_API_KEY` / `NGC_API_KEY` | — | bearer token for the Nebius endpoint |
| `MCP_HOST` | `127.0.0.1` | MCP bind host (keep localhost — Inv #10) |
| `IMAGING_PORT` / `XUE_PORT` | `9901` / `9902` | MCP server ports |
| `MCP_TRANSPORT` | `streamable-http` | `streamable-http` or `stdio` (Inspector) |
| `TIER1_DEVICE` / `TIER2_DEVICE` | `cpu` / auto | torch devices for Xue / MONAI |
| `CHECKPOINT_DIR` | `./checkpoints` | where MONAI bundles are cached |
| `MAX_SELF_CHECK_ITER` | `3` | critic↔reviser loop cap |
| `CRYCHIC_OVERLAY_DIR` | `./.crychic_overlays` | rendered finding-card PNGs |

---

## Out of Scope (do not re-add)

PET / amyloid input, **MYGO-Centiloid** (amyloid quantification), and **MUJICA**
(EPVS segmentation) are deliberately out of scope (CLAUDE.md §8). The pipeline inputs
are clinical-note + T1 (± FLAIR); the MRI signal fed to the Xue model is the T1's own
precomputed SwinUNETR embedding, not a separate scan.

---

## Team

- **Jimmy / Haozhe Jia** (lead) — BU BIL imaging AI; architecture & pitch
- **Jessie / Yujie Hu** — BU SPH; clinical NLP & the Tier-1 wrapper; problem statement
- **Zijiang Zhao** — BU MS AI / UConn CS; full-stack; OpenClaw integration & UX
- **Shixuan He** — NUU ML / UW-Madison capstone; reasoner prompts & overlays

---

## Key References

- Xue et al., *Nature Medicine* 2024 — AI-based differential diagnosis of dementia etiologies (clinical model)
- NIA-AA 2018 — A/T/(N) biological framework; structural neurodegeneration criteria
- Evans 1942 — ventricular (Evans) index (here approximated as an Evans-like index)
- Fazekas et al. 1987 / Wardlaw STRIVE — white-matter-hyperintensity grading & small-vessel-disease standards
- MONAI wholeBrainSeg (UNEST, 133-label whole-brain parcellation)

---

*v0.4 · Built for the Healthcare Agentic AI Hackathon, Harvard Innovation Labs.*
