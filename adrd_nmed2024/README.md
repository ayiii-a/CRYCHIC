# ADRD nmed2024 tool

A self-contained, NemoClaw-callable wrapper around the **Xue 2024** multimodal
dementia model (*"AI-based differential diagnosis of dementia etiologies on
multimodal data"*, Nature Medicine 2024; `vkola-lab/nmed2024`).

```
INPUT   structured patient cohort info (raw UDS clinical variables)
        + optionally one T1w MRI (skull-stripped, MNI-registered)
OUTPUT  predictions : probability for each of 13 diagnostic labels
                      (cognitive stage NC/MCI/DE + 10 etiologies)
        heatmap     : a Feature × label attribution matrix (PNG + CSV),
                      in the paper's annotated-viridis style
```

This folder is **fully self-contained and meant to be migrated** into the
NemoClaw agentic-flow system as-is — including the SwinUNETR SSL encoder, so the
raw-NIfTI → embedding path works out of the box.

> **Large files:** `assets/` holds the 20 MB fusion checkpoint and the 393 MB
> SwinUNETR encoder. When committing this folder to git, track `assets/*.pt`
> with **Git LFS** (`git lfs track "assets/*.pt"`).

---

## Folder layout

```
adrd_nmed2024/
├── README.md                  # this file
├── requirements.txt           # runtime deps (vendored adrd pkg is NOT pip-installed)
├── adrd_tool.py               # the tool: ADRDTool, run_adrd_tool(), TOOL_SPEC, CLI
├── mapping.py                 # verified UDS → model-input mapping (vendored)
├── demo.sh                    # runnable demo
├── examples/
│   └── example_clinical.json  # an AD-profile patient (raw UDS variables)
└── assets/
    ├── adrd/                          # vendored adrd package (inference code)
    ├── ckpt_swinunetr_stripped_MNI.pt # fusion checkpoint (20 MB)
    ├── model_swinvit.pt               # SwinUNETR SSL encoder (393 MB; raw-NIfTI path)
    └── input_meta_info.csv            # feature schema for the UDS mapping
```

---

## Install

```bash
pip install -r requirements.txt
```

`matplotlib`/`seaborn` are only needed to render the heatmap **PNG**; without
them the tool still returns the full attribution matrix as data and writes the
**CSV**. `monai`/`nibabel`/`einops` are only needed for the raw-NIfTI → embedding
path.

---

## Usage

### 1. Python object (keep the model warm in a long-lived agent)
```python
from adrd_tool import ADRDTool
tool = ADRDTool()                       # loads the checkpoint once
result = tool.analyze(
    clinical={"NACCAGE": 80, "SEX": 0, "EDUC": 13,
              "NACCMMSE": 22, "NACCNE4S": 1, "HACHIN": 1},
    mri="scan.nii.gz",                  # optional; .nii.gz or precomputed .npy
    out_dir="out/",                     # optional; writes the heatmap here
)
```

### 2. Stateless function (lazy singleton — JSON in / JSON out)
```python
from adrd_tool import run_adrd_tool, TOOL_SPEC
result = run_adrd_tool(clinical="examples/example_clinical.json", out_dir="out/")
```

### 3. CLI / subprocess (agent shells out, reads JSON from stdout)
```bash
python adrd_tool.py --clinical examples/example_clinical.json --out out/
python adrd_tool.py --clinical '{"NACCAGE":71,"NACCMMSE":22,"SEX":1}' --no-explain
python adrd_tool.py --clinical case.json --mri scan.nii.gz --out out/
```

### Demo
```bash
./demo.sh                                   # clinical-only on the bundled example
./demo.sh python3 ./demo_out scan.npy       # add an MRI embedding / .nii.gz
```

---

## NemoClaw integration

`adrd_tool.py` exposes `TOOL_SPEC`, a function-calling schema the agent registers:

```python
from adrd_tool import TOOL_SPEC, run_adrd_tool
# register TOOL_SPEC with the agent; route invocations to run_adrd_tool(**args)
```

`TOOL_SPEC["input_schema"]` ⇒ `{clinical (required), mri, out_dir}`.

---

## Input contract

- **clinical**: a dict / JSON file / single-row CSV of **raw UDS variable names**
  (e.g. `NACCAGE`, `NACCMMSE`, `SEX`, `NACCNE4S`, `EDUC`, `HACHIN`). The tool
  applies the section-prefix + value-remap mapping internally
  (see `mapping.py`). Any `ID` / label columns are ignored. Keys that are not
  recognized UDS variables are silently dropped.
- **mri** (optional): a skull-stripped, MNI-registered T1w `.nii/.nii.gz`, **or**
  a precomputed `(1, 768, 4, 4, 4)` SwinUNETR embedding `.npy`. The tool does
  **not** run SynthStrip/FLIRT — pass an already-preprocessed volume.

## Output contract

```jsonc
{
  "status": "ok",
  "input": { "id": ..., "n_clinical_features": 5, "mri": null, "imaging_used": false },
  "predictions": {
    "all":      { "NC": .29, "MCI": .42, "DE": .63, "AD": .59, ... },   // 13 labels
    "stage":    { "top": "DE", "probs": { "NC":.29, "MCI":.42, "DE":.63 } },
    "etiology": { "top": "AD", "probs": { "AD":.59, "LBD":.46, ... } }
  },
  "heatmap": {
    "kind": "feature_x_label_attribution",
    "rows": ["NACCAGE","EDUC","NACCMMSE","NACCNE4S","HACHIN","MRI (img)"],
    "columns": ["NC","MCI","DE","AD","LBD","VD","PRD","FTD","NPH","SEF","PSY","TBI","ODE"],
    "values": [[...], ...],                       // signed Δ-probability matrix
    "files": { "csv": "out/attribution_heatmap.csv", "png": "out/attribution_heatmap.png" }
  },
  "caveats": [ ... ]                              // present only when relevant
}
```

### The heatmap
A **Feature × 13-label attribution matrix** (paper `output.png` style: annotated
cells, viridis, gridlines, labels on top). Each cell is a leave-one-out
**occlusion delta**:

```
value[feature, label] = P(full)[label] − P(without that feature)[label]
```

`> 0` ⇒ the feature pushes that diagnosis **up**; `< 0` ⇒ down. When an MRI is
supplied, a single `MRI (img)` row reports `P(with MRI) − P(clinical-only)` per
label. Computed in one batched forward pass.

---

## Asset resolution (for migration)

Each asset is resolved as: **env var → bundled `assets/` → original dev path**.
After migrating this folder, set the env vars (or keep the bundled defaults):

| env var             | what                         | bundled? |
|---------------------|------------------------------|----------|
| `ADRD_CKPT`         | fusion checkpoint `.pt`      | ✅ (20 MB) |
| `ADRD_META`         | `input_meta_info.csv`        | ✅ |
| `ADRD_PKG_DIR`      | dir containing `adrd/`       | ✅ |
| `ADRD_SWIN_WEIGHTS` | SwinUNETR SSL weights (393 MB) | ✅ |

The SwinUNETR weights (`model_swinvit.pt`) are bundled and loaded **lazily** —
only when input is a **raw `.nii.gz`** (to embed it on the fly). If you always
pass a precomputed `.npy` embedding, they are never loaded and could be dropped
to slim the folder. Upstream source: MONAI
(`Project-MONAI/MONAI-extra-test-data` → `model_swinvit.pt`).

---

## Important caveats

- **Embeddings mode.** The fusion checkpoint is `img_net='SwinUNETREMB'`: it
  consumes a precomputed `(1, 768, 4, 4, 4)` SwinUNETR embedding, **not** raw
  voxels — there is no Swin backbone inside it. So the imaging signal enters as
  one embedding vector, and the heatmap's `MRI (img)` row is a modality-level
  contribution, not a voxel saliency map.
- **Base SSL encoder.** The bundled `.npy`-embedding path / the on-the-fly
  embedder use the public MONAI SSL SwinViT, **not** the authors' fine-tuned
  encoder. Treat imaging magnitudes as approximate; clinical predictions are
  unaffected.
- **UDS mapping is reconstructed** from the checkpoint + `input_meta_info.csv`
  (functionally verified), not the authors' private `nacc_variable_mappings.pkl`.
  Value-remap order may differ slightly from training.
- **Not a medical device.** Research/demo use only.

---

## Provenance & citation

Model and `adrd` package: `vkola-lab/nmed2024` (GPL/▶ see upstream LICENSE).
The UDS mapping and tool wrapper were verified against the CRYCHIC OASIS-3 demo
cases.

```bibtex
@article{xue2024ai,
  title={AI-based differential diagnosis of dementia etiologies on multimodal data},
  author={Xue, Chonghua and Kowshik, Sahana S and Lteif, Diala and others},
  journal={Nature Medicine}, year={2024},
  doi={10.1038/s41591-024-03118-z}
}
```
