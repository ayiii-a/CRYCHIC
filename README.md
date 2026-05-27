# CRYCHIC-MCP

> **CRYCHIC**: Multi-modal dementia clinical decision support, exposed as an MCP server.
> Submission for the Healthcare Agentic AI Hackathon (Harvard Innovation Labs, 2026-05-30).

---

## What This Is

`crychic-mcp` is the **MCP server** that wraps the entire CRYCHIC pipeline (4 imaging/clinical models + 3 Nemotron LLM steps + a self-check loop) behind **3 clean tools** that OpenClaw / Nemotron orchestrators can call.

- **Outside the server**: a clean MCP interface — start a case, poll status, retrieve evidence.
- **Inside the server**: async pipeline orchestration, parallel imaging inference, rule-based pattern matching, and an LLM self-critique loop for CDS compliance.

This is a **Clinical Decision Support (CDS) tool, not a diagnostic device**. The clinician makes the final decision; CRYCHIC only organizes evidence.

---

## Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────┐
│ UI Interface                                                 │
└──────────────┬───────────────────────────────────────────────┘
               │ file upload event
               ▼
┌──────────────────────────────────────────────────────────────┐
│ OpenClaw / Nemotron orchestrator                             │
│   ├─ MCP Client  ──── stdio/SSE ────► crychic-mcp            │
│   └─ HTTP Client ──── OpenAI API ───► nemotron-nim            │
└────┬─────────────────────────────────────────────┬───────────┘
     │                                             │
     ▼                                             ▼
┌──────────────────────┐                 ┌──────────────────────┐
│ Container: crychic-mcp│                │ Container: nemotron  │
│ (this repo)           │                │ (NVIDIA NIM)         │
│                       │                │                      │
│ MCP Server            │                │ Nemotron-Nano-9B     │
│  ├─ run_crychic_pipe… │                │ OpenAI-compatible    │
│  ├─ get_pipeline_stat…│                │ port 8000            │
│  └─ get_case_evidence │                │                      │
│                       │                │                      │
│ Pipeline (internal):  │                │                      │
│  ├─ Tier-1: Xue 2024  │                │                      │
│  ├─ Router (LLM ──────┼───────────────►│                      │
│  ├─ Centiloid (MYGO)  │                │                      │
│  ├─ EPVS (MUJICA)     │                │                      │
│  ├─ Anatomy (MONAI)   │                │                      │
│  ├─ Aggregator (rules)│                │                      │
│  └─ Reasoner + Critic │                │                      │
│       loop (LLM ──────┼───────────────►│                      │
└──────────────────────┘                  └─────────────────────┘
        GPU 0 (shared)
```

**Key design choice**: the LLM is called *from inside* the MCP tool, not from outside. This means Nemotron never has to do tool-calling on its own — we keep the small model on plain text-in / text-out tasks where it is reliable.

---

## The 3 MCP Tools

### 1. `run_crychic_pipeline`

Fire the full pipeline. Returns immediately with a `case_id`; the pipeline runs async in the background.

**Input**:
```json
{
  "pet_path": "/data/case_a/PET.nii.gz",
  "t1_path": "/data/case_a/T1.nii.gz",
  "clinical_text": "78F, 6mo memory decline, APOE ε4/ε3, MMSE 22 ...",
  "tracer": "PiB"
}
```

**Output**:
```json
{
  "case_id": "case_a1b2c3d4",
  "status": "started",
  "message": "Pipeline running async. Poll get_pipeline_status."
}
```

### 2. `get_pipeline_status`

Poll the current execution stage. Used by the orchestrator to push progress updates to Slack.

**Input**: `{ "case_id": "case_a1b2c3d4" }`

**Output**:
```json
{
  "case_id": "case_a1b2c3d4",
  "stage": "imaging",
  "completed_tools": ["centiloid", "mujica"],
  "elapsed_seconds": 14.3,
  "error": null
}
```

**Stage sequence**:
`initialized` → `screening` → `routing` → `imaging` → `aggregating` → `reasoning` → `self_check_attempt_1` → (`revising_attempt_1` → `self_check_attempt_2` ...) → `self_check_passed` → `complete` | `failed`

### 3. `get_case_evidence`

Retrieve structured intermediate evidence. Used for follow-up questions ("show me the EPVS breakdown") without re-running the pipeline.

**Input**:
```json
{
  "case_id": "case_a1b2c3d4",
  "fields": ["tier1", "tier2", "pattern", "conflicts", "report"]
}
```

**Output**: structured JSON with all requested fields. Omit `fields` to get everything.

---

## Pipeline Stages (What Happens Inside)

### Stage 1 — Tier-1 Clinical Screening (pure Python, no LLM)

Runs the Xue 2024 / Kola Lab model on the clinical text. Outputs 13 independent probabilities:

- Cognitive stage: NC, MCI, DE
- Etiologies: AD, LBD, VD, PRD, FTD, NPH, SEF, PSY, TBI, ODE

These probabilities are **not mutually exclusive** — a patient can have elevated P(AD), P(VD), and P(MCI) simultaneously, reflecting real-world comorbidity.

### Stage 2 — Router (Nemotron LLM + Python rules)

Nemotron generates a **rationale** explaining which imaging tools should be invoked. The actual triggering is done by **hard-coded clinical rules** (Section 2.3 of the clinical brief):

| Trigger | Tool |
|---|---|
| `P(AD) ≥ 0.30` or `P(MCI) ≥ 0.40` | MYGO-Centiloid (amyloid PET quantification) |
| `P(VD) ≥ 0.20` or `P(AD) ≥ 0.50` | MUJICA (EPVS segmentation) |
| Always | MONAI wholeBrainSeg (anatomy) |

**Why split this way**: the LLM produces the *clinical reasoning text* (which the orchestrator can display to the clinician), but the *execution path* is deterministic and auditable. This is the core CDS design principle.

### Stage 3 — Parallel Imaging (Python `asyncio.gather`)

The selected imaging tools run in parallel:

- **MYGO-Centiloid**: 3D ResNet-18 + TracerNorm. Outputs Centiloid value, positivity (≥ 20, GAAIN standard), cortical SUVR overlay.
- **MUJICA**: 3D Attention U-Net with DiceFocal loss. Outputs total/BG/CSO EPVS volumes, distribution pattern, burden grade.
- **MONAI wholeBrainSeg**: hippocampal volumes, Z-score, ventricular index, dominant atrophy pattern.

All inference is local on a shared GPU. No PHI leaves the container.

### Stage 4 — Aggregator (pure Python rules)

Two pure-Python functions:

- `match_clinical_pattern(tier1, tier2)` — maps evidence to one of 6 predefined AD-spectrum patterns (Section 2.5). Returns a `ClinicalPattern` with `pattern_id`, `rationale`, and `supporting_evidence`.
- `detect_conflicts(tier1, tier2)` — surfaces evidence conflicts (Section 2.6) such as "amyloid-negative AD syndrome" or "subclinical amyloid". **Conflicts are surfaced, never silently overridden.**

### Stage 5 — Reasoner + Self-Check Loop (3 Nemotron calls in a loop)

This is the agentic core:

1. **Reasoner** generates a Markdown report from structured evidence, following the 6 CDS principles.
2. **Critic** checks the report against the 6 principles. Outputs `PASS` or a bulleted list of violations.
3. **Reviser** fixes the violations. Loop back to Critic.

Max 3 iterations. If the loop does not converge, the report is returned with a warning flag.

**Why this is genuinely agentic**: the LLM decides on its own when the report is good enough to ship. The number of iterations is not fixed.

---

## The 6 CDS Principles (Enforced by the Critic Loop)

| # | Principle | What the Critic Checks |
|---|---|---|
| 1 | Hedged language | No "diagnose", "recommend", or imperative verbs |
| 2 | CDS header/footer | Both disclaimers present |
| 3 | Traceable claims | Every numeric value has threshold + reference citation |
| 4 | Counter-evidence required | Differential & limitations section non-empty |
| 5 | Options not recommendations | `○` bullets, never numbered |
| 6 | Sign-off buttons | `[✓ Agree & sign] [✏️ Edit] [✗ Disagree]` in footer |

Only the signed version enters the medical record. The draft never auto-commits.

---

## Repository Layout

```
crychic-mcp/
├── CLAUDE.md                       # this file
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── crychic_mcp_server.py           # MCP entry point (3 tools)
└── crychic/
    ├── __init__.py
    ├── state.py                    # CaseStore + CaseState (in-memory, ephemeral)
    ├── schemas.py                  # Pydantic data contracts
    ├── llm.py                      # Nemotron HTTP client + 4 system prompts
    ├── pipeline.py                 # Main async orchestration
    ├── tier1_screening.py          # Xue 2024 wrapper
    ├── tier2_imaging.py            # MYGO-Centiloid / MUJICA / MONAI wrappers
    └── aggregator.py               # Pattern matching + conflict detection (pure rules)
```

---

## Quick Start

### Local Development (no Docker)

```bash
# 1. Start Nemotron NIM (in a separate terminal)
docker run --gpus all -p 8000:8000 \
  -e NGC_API_KEY=$NGC_API_KEY \
  nvcr.io/nim/nvidia/nemotron-nano-9b-v2:latest

# 2. Install deps
pip install -r requirements.txt

# 3. Drop checkpoints into ./checkpoints/
#    - mujica_attention_unet.pt
#    - mygo_centiloid_resnet18.pt
#    - monai_wholebrainseg.pt (or MONAI Bundle)

# 4. Run the MCP server (stdio mode)
NEMOTRON_URL=http://localhost:8000/v1/chat/completions \
  python crychic_mcp_server.py

# 5. Test with the MCP Inspector
npx @modelcontextprotocol/inspector python crychic_mcp_server.py
```

### Docker Compose (recommended for demo day)

```bash
export NGC_API_KEY=...
docker compose up --build
```

This brings up both `nemotron` and `crychic-mcp` containers on a shared network. The MCP server reaches Nemotron at `http://nemotron:8000`.

### Pre-cache the 3 demo cases

```bash
# Run each case once so the imaging model weights are warm in GPU memory
# and the case results are stored in CaseStore
python scripts/precache_demos.py --cases A B C
```

This is **critical** — cold GPU inference on the first request can take 60+ seconds, well past the 30-second demo budget.

---

## Configuration

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `NEMOTRON_URL` | `http://nemotron:8000/v1/chat/completions` | OpenAI-compatible endpoint |
| `NEMOTRON_MODEL` | `nvidia/nemotron-nano-9b-v2` | Model name (passed in request body) |
| `MCP_TRANSPORT` | `stdio` | `stdio` or `sse` |
| `MCP_SSE_PORT` | `3000` | Port for SSE mode |
| `CHECKPOINT_DIR` | `/app/checkpoints` | Where imaging model weights live |
| `MAX_SELF_CHECK_ITER` | `3` | Critic-Reviser loop cap |

---

## Demo Day Sequence (Reference Timing)

| Time | Stage | What the Clinician Sees in Slack |
|---|---|---|
| T+0s | (upload) | "📋 Case `case_a1b2` received. Analyzing..." |
| T+1s | `screening` | "🔍 Clinical screening..." |
| T+2s | `routing` | (router rationale streams in) |
| T+3s | `imaging` | "⏳ Running 3 imaging tools in parallel..." |
| T+10s | `imaging` | "✅ Centiloid done" |
| T+15s | `imaging` | "✅ MUJICA done" |
| T+20s | `imaging` | "✅ MONAI done" |
| T+21s | `aggregating` | "🧩 Matching clinical pattern..." |
| T+23s | `reasoning` | "📝 Drafting report..." |
| T+27s | `self_check` | "🔍 Validating against 6 CDS principles..." |
| T+30s | `complete` | (full Markdown report + sign-off buttons) |

If a clinician then asks a follow-up like "Why pattern 4 and not pattern 2?", the orchestrator calls `get_case_evidence` and lets Nemotron explain — **no pipeline re-run needed**.

---

## Why This Is Agentic (For the Pitch)

This system is a **bounded agentic system**, not a fully autonomous agent. The agency lives at three explicit points:

1. **Router rationale** — the LLM generates clinical reasoning for tool selection (executed deterministically by rules for auditability).
2. **Report synthesis** — the LLM composes a clinical narrative from structured evidence.
3. **Self-critique loop** — the LLM decides on its own when the report meets CDS compliance, iterating until convergence.

Fully autonomous agents are an **anti-pattern in clinical decision support** — they violate FDA CDS exemption requirements around traceability, reproducibility, and clinician oversight. CRYCHIC's design is the *correct* shape for healthcare AI in 2026, not a technical compromise.

The differentiator: **"AI shows you the evidence — you decide,"** not "AI tells you the diagnosis."

---

## Limitations (Explicitly Surfaced)

The system itself surfaces these in every report when relevant:

1. No Tau PET input — cannot distinguish Vogel 2021 tau trajectory subtypes.
2. No DaTscan / α-syn SAA input — LBD comorbidity can only be suggested, not confirmed.
3. No SWI segmentation in v0.3 — CAA signature is inferred via MUJICA CSO-EPVS pattern; Boston v2.0 scoring still requires manual SWI review.
4. Single timepoint — pseudodementia (Pattern 6) requires longitudinal follow-up to fully exclude.
5. Tier-1 model trained on NACC — distribution shift exists when applied to OASIS-3 or other cohorts.

---

## Team

- **Jimmy / Haozhe Jia** (lead) — BU BIL imaging AI; MUJICA & MYGO-Centiloid; architecture & pitch
- **Jessie / Yujie Hu** — BU SPH; clinical NLP & Tier-1 wrapper; pitch problem statement
- **Zijiang Zhao** — BU MS AI / UConn CS; full-stack; OpenClaw integration & UX
- **Shixuan He** — NUU ML / UW-Madison capstone; Reasoner prompt engineering & overlays

MYGO-Centiloid is under review at the vkola lab (Xue 2024 authors).
MUJICA is from the BU BIL (Koo lab) EPVS research line.

---

## Key References

- Xue et al., *Nature Medicine* 2024 — AI-based differential diagnosis of dementia etiologies
- Klunk et al. 2015 — Centiloid standardization
- NIA-AA 2018 — A+T+ biological framework
- Iliff 2012, Nedergaard 2013 — Glymphatic system
- Wardlaw STRIVE — small vessel disease imaging standard
- Boston v2.0 — CAA imaging diagnosis

---

*v0.3 · 2026-05-27 · Built for Healthcare Agentic AI Hackathon, Harvard Innovation Labs.*
