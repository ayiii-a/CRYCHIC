# Label-Symptom Alignment in CRYCHIC

This note answers the practical question: when CRYCHIC shows a high model label, does that label mechanically line up with the patient's symptoms or impairment pattern?

Short answer: yes, conceptually. The 13 labels are designed to map to clinically meaningful cognitive states and dementia etiologies. But a high label is not a diagnosis by itself. It is a signal that should be checked against symptoms, imaging, biomarkers, and clinician judgment.

CRYCHIC should be explained as evidence organization, not clinical decision-making.

## How The Labels Work

The Tier-1 model outputs 13 probabilities.

The first 3 labels describe severity of cognitive impairment:

| Label | Meaning | Symptom / function alignment |
|---|---|---|
| `NC` | Normal cognition | No clear objective cognitive syndrome; function mostly intact |
| `MCI` | Mild cognitive impairment | Objective cognitive decline, but basic independence mostly preserved |
| `DE` | Dementia | Cognitive decline severe enough to interfere with daily function |

The other 10 labels describe possible causes or contributors:

| Label | Meaning | Symptom / impairment pattern it should raise |
|---|---|---|
| `AD` | Alzheimer's disease | Episodic memory loss, repeated questions, disorientation, later functional decline |
| `LBD` | Lewy body dementia / Parkinson's disease dementia | Fluctuating cognition, visual hallucinations, REM sleep behavior disorder, parkinsonism, visuospatial or attention deficits |
| `VD` | Vascular dementia / vascular brain injury | Executive dysfunction, slowed processing, attention problems, gait issues, stepwise decline, stroke or vascular risk history |
| `PRD` | Prion disease | Rapidly progressive dementia, myoclonus, ataxia, visual symptoms, urgent decline over weeks to months |
| `FTD` | Frontotemporal dementia | Behavior change, disinhibition, apathy, loss of empathy, compulsions, progressive language impairment, executive dysfunction |
| `NPH` | Normal pressure hydrocephalus | Gait disturbance, urinary urgency/incontinence, slowed thinking, ventriculomegaly |
| `SEF` | Systemic/environmental factors | Delirium, infection, metabolic disease, medication toxicity, sleep apnea, substance use, fluctuating or reversible symptoms |
| `PSY` | Psychiatric conditions | Depression, anxiety, psychosis, PTSD, poor concentration, slowed processing, pseudodementia-like presentation |
| `TBI` | Traumatic brain injury | Attention/executive dysfunction, memory problems, irritability, sleep issues, trauma history |
| `ODE` | Other dementia etiologies | Atypical course, seizures, movement disorder, focal signs, genetic syndrome, tumor/structural lesion, or unclear fit |

## What A High Label Means Mechanically

A high label means the input features resemble cases that the model learned to associate with that clinical bucket.

In this project, labels should be interpreted like this:

- High `MCI` means the model sees impairment below dementia-level functional loss.
- High `DE` means the model sees dementia-level impairment.
- High `AD`, `VD`, `LBD`, etc. means the model sees evidence compatible with that possible contributor.
- Multiple high etiology labels can coexist. This is expected because dementia is often mixed.

The labels are not mutually exclusive. For example:

```text
High DE + high AD + moderate VD
```

This should be explained as dementia-level impairment with an Alzheimer-pattern signal and possible vascular contribution, not as two separate confirmed diagnoses.

## Where CRYCHIC Uses The Labels

CRYCHIC uses the labels in three main ways.

### 1. Explain the clinical state

The stage labels answer:

```text
How impaired does the patient appear?
NC vs MCI vs DE
```

This is symptom/function alignment.

### 2. Route additional tools

The router runs the free T1 structural baseline whenever a T1 is present, and makes
one real decision about the extra, modality-gated axis (FLAIR → WMH/Fazekas):

| Trigger | Imaging axis (etiology) |
|---|---|
| T1 present | Free structural baseline: hippocampal Z (AD), automated Evans-like index (NPH), frontotemporal lobar Z (FTD) |
| `P(VD) >= 0.20` **and FLAIR present** | WMH → Fazekas grade (vascular burden) |
| PRD / SEF / PSY / TBI / LBD (and ODE) | Abstain — no off-the-shelf structural correlate; rely on clinical features |

This means a high label does not directly produce a final answer. It triggers evidence gathering.

Example:

```text
High AD  -> ask whether hippocampal atrophy (Z < -1.5) supports the AD-like signal.
High NPH -> ask whether the automated Evans-like index (> 0.30) shows ventriculomegaly.
High FTD -> ask whether frontotemporal volume loss is disproportionate to global.
High VD  -> if FLAIR is available, ask whether WMH burden (Fazekas >= 2) supports a vascular contribution.
High MCI -> gather early-stage structural evidence without calling it dementia.
```

### 3. Detect mismatches

The aggregator explicitly checks for cases where the label and downstream evidence do not line up.

Examples:

| Mismatch | Why it matters |
|---|---|
| High `AD` but no hippocampal atrophy | Could be early/atypical AD, an AD mimic, or limits of single-timepoint volumetry |
| Hippocampal atrophy but low `AD` | Could be a non-AD process (LATE, hippocampal sclerosis) or age-related change |
| Ventriculomegaly but low `NPH` | May reflect atrophy ex vacuo rather than hydrocephalus |
| Dementia-level `DE` but benign structural imaging | Symptoms may exceed imaging burden; consider psychiatric, systemic, medication, or an unassessed axis (e.g. vascular without FLAIR) |

This is important for judge questions: CRYCHIC does not hide label-evidence conflicts. It surfaces them.

## What The Labels Do Not Prove

A high label does not prove:

- the patient has that disease;
- the symptom is caused only by that etiology;
- a treatment should be started;
- the model is clinically correct;
- the evidence is sufficient for a medical record diagnosis.

Instead, a high label means:

```text
This clinical pattern is plausible enough to organize evidence around it.
```

That is the safe CDS framing.

## Best Explanation For The Demo

Use this wording:

> The labels are clinically meaningful buckets. `MCI` and `DE` describe impairment severity, while labels like `AD`, `VD`, `LBD`, and `FTD` describe possible contributors. A high label does not decide the case. It tells the agent what evidence to gather next and what conflicts to surface for clinician review.

If judges ask whether the label matches the symptom:

> Yes, at the level of clinical pattern matching. For example, high `AD` should correspond to an amnestic Alzheimer-like pattern, high `VD` to executive/slowed-processing and vascular risk patterns, high `LBD` to hallucinations/fluctuations/parkinsonism, and high `NPH` to gait-urinary-cognitive triad. But CRYCHIC treats those as hypotheses and checks them against imaging and biomarkers before drafting the report.

## Safety Position

The safest way to describe the project:

- The model labels are hypotheses, not conclusions.
- The agent routes tools based on deterministic rules.
- The report cites evidence and thresholds.
- Conflicts are surfaced instead of resolved silently.
- A clinician must review, edit, agree, or disagree before anything is used clinically.

This is why the label-to-symptom mapping is useful without becoming unsafe.
