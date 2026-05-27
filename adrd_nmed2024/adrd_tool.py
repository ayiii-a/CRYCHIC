"""
ADRD diagnosis tool — a NemoClaw-callable wrapper around the Xue 2024
(Nature Medicine) multimodal dementia model.

    INPUT   structured patient cohort info (raw UDS clinical variables)
            + optionally one T1w MRI (skull-stripped, MNI-registered)
    OUTPUT  - predictions : probability for each of the 13 diagnostic labels,
                            split into cognitive stage (NC/MCI/DE) + etiology
            - heatmap     : a Feature x 13-label attribution matrix in the
                            paper's output.png style (annotated, viridis),
                            written as .png + .csv and returned inline

The model runs in embeddings mode (img_net='SwinUNETREMB'): it consumes a
precomputed (1, 768, 4, 4, 4) SwinUNETR embedding, not raw voxels. Attributions
are leave-one-out occlusion deltas (P(full) - P(without feature)); the MRI is a
single "MRI (img)" row = P(with MRI) - P(clinical-only).

Three call styles
-----------------
1. Python object (keep the model warm in a long-lived agent):
       from adrd_tool import ADRDTool
       tool = ADRDTool()
       result = tool.analyze(clinical={...}, mri="scan.nii.gz", out_dir="out/")

2. Stateless function (lazy singleton, JSON in / JSON out):
       from adrd_tool import run_adrd_tool
       result = run_adrd_tool(clinical="case.json", out_dir="out/")

3. CLI / subprocess (agent shells out, reads JSON from stdout):
       python adrd_tool.py --clinical case.json --mri scan.nii.gz --out out/

``TOOL_SPEC`` (bottom of file) is the function-calling schema to register.

Asset resolution (each overridable by env var; falls back to the bundled
``assets/`` dir, then to the original CRYCHIC dev locations):
    ADRD_CKPT          checkpoint .pt
    ADRD_META          input_meta_info.csv
    ADRD_PKG_DIR       dir containing the `adrd` package
    ADRD_SWIN_WEIGHTS  SwinUNETR SSL weights (only for raw-NIfTI input; 393 MB)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
_ASSETS = _HERE / "assets"
# Original dev locations, used only as a last-resort fallback before migration.
_DEV_HF = Path("~/crychic/models/nmed2024-hf").expanduser()
_DEV_WEIGHTS = Path("~/crychic/models/weights/model_swinvit.pt").expanduser()


def _resolve(env: str, *candidates: Path) -> Path:
    """First of: $env, then each candidate that exists; else the first candidate."""
    if os.environ.get(env):
        return Path(os.environ[env]).expanduser()
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


CHECKPOINT_PATH = _resolve("ADRD_CKPT",
                           _ASSETS / "ckpt_swinunetr_stripped_MNI.pt",
                           _DEV_HF / "ckpt_swinunetr_stripped_MNI.pt")
META_PATH = _resolve("ADRD_META",
                     _ASSETS / "input_meta_info.csv",
                     _DEV_HF / "data" / "input_meta_info.csv")
PKG_DIR = _resolve("ADRD_PKG_DIR", _ASSETS, _DEV_HF)
SWIN_WEIGHTS = _resolve("ADRD_SWIN_WEIGHTS", _ASSETS / "model_swinvit.pt", _DEV_WEIGHTS)

# Make `import adrd` resolve, and import the vendored mapping.
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import mapping as M  # noqa: E402  (vendored; also applies the torch.load 2.6+ patch)

LABELS = M.LABEL_COLS
STAGE_LABELS = ["NC", "MCI", "DE"]
ETIOLOGY_LABELS = [l for l in LABELS if l not in STAGE_LABELS]
IMG_KEY = "img_MRI_T1_1"
_PREFIXES = ("his_", "bat_", "exam_", "updrs_", "med_", "gds_",
             "cvd_", "npiq_", "ph_", "faq_", "apoe_", "img_")


@contextlib.contextmanager
def _quiet():
    """Silence the adrd library's internal stdout/stderr (device prints, tqdm)."""
    with open(os.devnull, "w") as null, \
            contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
        yield


def _to_py(x):
    return x.item() if isinstance(x, (np.floating, np.integer)) else x


def _deprefix(key: str) -> str:
    for p in _PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


class ADRDTool:
    """Loads the ADRD model once and answers many patients."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        self.src_modalities = state["src_modalities"]
        self.mapping = M.build_mapping(self.src_modalities, pd.read_csv(META_PATH))
        import adrd
        with _quiet():
            self.model = adrd.model.ADRDModel.from_ckpt(str(CHECKPOINT_PATH), device=device)
        self._swin = None  # built lazily only for raw-NIfTI input

    # ----------------------------- public API ----------------------------- #
    def analyze(
        self,
        clinical: dict | str | Path,
        mri: str | Path | None = None,
        explain: bool = True,
        out_dir: str | Path | None = None,
        cmap: str = "viridis",
    ) -> dict:
        """Run one patient -> JSON-serializable dict (predictions + heatmap)."""
        caveats: list[str] = []
        raw = self._load_clinical(clinical)
        feats = M.convert_dictionary(raw, self.mapping, self.src_modalities)
        if not feats:
            caveats.append("No clinical features mapped — keys must be raw UDS "
                           "variable names; model would return its prior.")

        emb, mri_img = None, None
        if mri is not None:
            emb, mri_img, c = self._embed_mri(mri)
            caveats += c

        x = dict(feats)
        if emb is not None:
            x[IMG_KEY] = emb
        with _quiet():
            proba = {k: float(v) for k, v in self.model.predict_proba([x])[1][0].items()}

        result: dict[str, Any] = {
            "status": "ok",
            "input": {
                "id": raw.get("ID") or raw.get("id"),
                "n_clinical_features": len(feats),
                "mri": str(mri) if mri is not None else None,
                "imaging_used": emb is not None,
            },
            "predictions": self._format_predictions(proba),
        }
        if explain:
            result["heatmap"] = self._attribution_matrix(feats, emb, out_dir, cmap)
        if caveats:
            result["caveats"] = caveats
        return result

    # --------------------------- result shaping --------------------------- #
    def _format_predictions(self, proba: dict) -> dict:
        stage = {k: proba[k] for k in STAGE_LABELS}
        etio = {k: proba[k] for k in ETIOLOGY_LABELS}
        return {
            "all": proba,
            "stage": {"top": max(stage, key=stage.get), "probs": stage},
            "etiology": {"top": max(etio, key=etio.get), "probs": etio},
        }

    # ------------------ Feature x 13-label attribution heatmap ------------- #
    def _attribution_matrix(self, feats: dict, emb, out_dir, cmap: str) -> dict:
        """Leave-one-out occlusion delta for every (feature, label) pair.

        value[feature, label] = P(full)[label] - P(without feature)[label].
        > 0 means the feature pushed that label up. If MRI is present, a single
        "MRI (img)" row = P(with MRI) - P(clinical-only). One batched forward.
        """
        keys = list(feats.keys())
        base = dict(feats)
        if emb is not None:
            base[IMG_KEY] = emb

        batch = [base]
        for drop in keys:  # leave-one-feature-out, keep imaging fixed
            v = {k: val for k, val in feats.items() if k != drop}
            if emb is not None:
                v[IMG_KEY] = emb
            batch.append(v)
        if emb is not None:  # last entry = clinical-only -> the MRI row baseline
            batch.append(dict(feats))

        with _quiet():
            probas = self.model.predict_proba(batch)[1]
        full = probas[0]

        rows, row_labels = [], []
        for i, key in enumerate(keys, start=1):
            row_labels.append(_deprefix(key))
            rows.append([float(full[l] - probas[i][l]) for l in LABELS])
        if emb is not None:
            clinical_only = probas[-1]
            row_labels.append("MRI (img)")
            rows.append([float(full[l] - clinical_only[l]) for l in LABELS])

        mat = pd.DataFrame(rows, index=row_labels, columns=LABELS)
        info: dict[str, Any] = {
            "kind": "feature_x_label_attribution",
            "note": "occlusion delta P(full) - P(without feature); MRI row = "
                    "P(with MRI) - P(clinical-only).",
            "rows": row_labels,
            "columns": LABELS,
            "values": [[round(v, 4) for v in r] for r in rows],
        }
        if out_dir is not None and row_labels:
            info["files"] = self._write_heatmap(mat, out_dir, cmap)
        return info

    def _write_heatmap(self, mat: pd.DataFrame, out_dir, cmap: str) -> dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        csv_path = out / "attribution_heatmap.csv"
        mat.round(4).to_csv(csv_path)
        files["csv"] = str(csv_path)
        try:  # PNG is best-effort (needs seaborn + matplotlib)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
            vabs = float(np.abs(mat.values).max()) or 1.0
            h = max(2.4, 0.4 * len(mat) + 1.2)
            fig, ax = plt.subplots(figsize=(7.5, h), dpi=200)
            sns.heatmap(mat, annot=True, fmt=".2f", cmap=cmap, vmin=-vabs, vmax=vabs,
                        linewidths=0.5, linecolor="black", cbar=True,
                        cbar_kws={"label": "Δ probability"}, ax=ax)
            ax.xaxis.tick_top()
            ax.set_title("Feature → diagnosis attribution", pad=24)
            plt.setp(ax.get_yticklabels(), rotation=0)
            fig.tight_layout()
            png_path = out / "attribution_heatmap.png"
            fig.savefig(png_path, bbox_inches="tight")
            plt.close(fig)
            files["png"] = str(png_path)
        except Exception as e:  # never break the tool over a plotting dep
            files["png_error"] = repr(e)
        return files

    # ------------------------------ loaders ------------------------------- #
    @staticmethod
    def _load_clinical(clinical) -> dict:
        if isinstance(clinical, dict):
            return clinical
        p = Path(clinical)
        if p.suffix.lower() == ".json":
            return json.loads(p.read_text())
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p).iloc[0].dropna().to_dict()
        raise ValueError(f"Unsupported clinical input: {clinical!r}")

    def _embed_mri(self, mri):
        p = Path(mri)
        caveats: list[str] = []
        if p.suffix == ".npy":
            return np.load(p).astype(np.float32), None, caveats
        caveats.append("MRI assumed skull-stripped + MNI-registered; this tool "
                       "does not run SynthStrip/FLIRT.")
        caveats.append("Embedding uses the base MONAI SSL SwinViT encoder, not the "
                       "authors' fine-tuned encoder — treat magnitudes as approximate.")
        emb = self._swin_embed(p)
        import nibabel as nib
        return emb, nib.load(str(p)), caveats

    def _swin_embed(self, nii_path: Path) -> np.ndarray:
        from monai.networks.nets import SwinUNETR
        from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
                                       Spacingd, CropForegroundd, Resized)
        if self._swin is None:
            if not Path(SWIN_WEIGHTS).exists():
                raise FileNotFoundError(
                    f"SwinUNETR weights not found at {SWIN_WEIGHTS}. Set "
                    "ADRD_SWIN_WEIGHTS, or pass a precomputed .npy embedding instead.")
            net = SwinUNETR(in_channels=1, out_channels=1, feature_size=48,
                            use_checkpoint=False)
            net.load_from(weights=torch.load(SWIN_WEIGHTS, map_location="cpu",
                                             weights_only=False))
            self._swin = net.eval().to(self.device)
        pre = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            CropForegroundd(keys=["image"], source_key="image"),
            Resized(keys=["image"], spatial_size=(128, 128, 128), mode="trilinear"),
        ])
        img = pre({"image": str(nii_path)})["image"]
        eps = torch.finfo(torch.float32).eps
        img = torch.nn.functional.relu((img - img.min()) / (img.max() - img.min() + eps))
        x = img.unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            hidden = self._swin.swinViT(x, self._swin.normalize)
        return hidden[4].cpu().numpy().astype(np.float32)  # (1,768,4,4,4)


# --------------------- stateless functional entrypoint --------------------- #
_SINGLETON: ADRDTool | None = None


def run_adrd_tool(clinical, mri=None, explain=True, out_dir=None,
                  cmap="viridis", device="cpu") -> dict:
    """Lazy-singleton wrapper so an agent can call without managing state."""
    global _SINGLETON
    if _SINGLETON is None or _SINGLETON.device != device:
        _SINGLETON = ADRDTool(device=device)
    return _SINGLETON.analyze(clinical=clinical, mri=mri, explain=explain,
                              out_dir=out_dir, cmap=cmap)


TOOL_SPEC = {
    "name": "adrd_diagnosis",
    "description": (
        "Differential dementia diagnosis from structured patient cohort data and an "
        "optional T1w MRI. Returns probabilities for 13 labels (cognitive stage "
        "NC/MCI/DE + 10 etiologies such as AD, LBD, VD, FTD) plus a Feature x label "
        "attribution heatmap (.png/.csv) explaining which clinical features and the "
        "MRI drove the prediction. Built on the Xue 2024 (Nature Medicine) ADRD model."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "clinical": {
                "type": "object",
                "description": "Raw UDS clinical variables, e.g. {'NACCAGE': 71, "
                               "'NACCMMSE': 22, 'SEX': 1, 'NACCNE4S': 1, 'EDUC': 16}.",
            },
            "mri": {
                "type": "string",
                "description": "Path to a skull-stripped, MNI-registered T1w .nii.gz, "
                               "or a precomputed .npy SwinUNETR embedding. Optional.",
            },
            "out_dir": {
                "type": "string",
                "description": "Directory to write the heatmap (.png/.csv). Optional.",
            },
        },
        "required": ["clinical"],
    },
}


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ADRD diagnosis tool (NemoClaw Stage 1).")
    ap.add_argument("--clinical", required=True,
                    help="JSON dict, single-row CSV, or inline JSON string.")
    ap.add_argument("--mri", default=None, help="Preprocessed .nii.gz or .npy embedding.")
    ap.add_argument("--out", dest="out_dir", default=None, help="Heatmap output dir.")
    ap.add_argument("--cmap", default="viridis", help="Heatmap colormap.")
    ap.add_argument("--no-explain", action="store_true", help="Skip the heatmap.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    clinical = args.clinical
    if clinical.strip().startswith("{"):
        clinical = json.loads(clinical)
    result = run_adrd_tool(clinical=clinical, mri=args.mri, out_dir=args.out_dir,
                           cmap=args.cmap, explain=not args.no_explain, device=args.device)
    print(json.dumps(result, indent=2, default=_to_py))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
