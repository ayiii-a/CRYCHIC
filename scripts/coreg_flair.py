#!/usr/bin/env python
"""Coregister a subject's FLAIR onto the MNI T1 grid → enables the VD/Fazekas axis.

The bundled OASIS-3 demo cohort ships T1 only, so the WMH/Fazekas (VD) check always
abstains. Once you have fetched a subject's FLAIR from OASIS-3 (controlled access —
see below), this script rigidly aligns it to the same skull-stripped MNI space the
T1 segmentation already uses and writes it where ``crychic.cases.find_flair`` looks
(``data/mri_prepro/{prefix}_FLAIR_MNI.nii.gz``). After that, ``run_one_case.py`` and
the web UI pick up the FLAIR automatically and ``wmh.fazekas`` runs for real.

Pipeline (FSL, the user-chosen FLIRT 6-dof rigid path)
------------------------------------------------------
  1. BET the native FLAIR (skull-strip) so cross-modal registration is robust.
  2. FLIRT  flair_brain → native stripped T1   (6 dof rigid, corratio)   = flair2t1.mat
  3. convert_xfm  flair2t1.mat ∘ {subject}_to_mni.mat                     = flair2mni.mat
  4. FLIRT  apply flair2mni.mat, -ref the stripped_MNI T1                 = FLAIR on MNI grid

When the native stripped T1 / ``_to_mni.mat`` are missing, it falls back to a single
direct FLIRT of the FLAIR onto the stripped_MNI T1.

Fetching the FLAIR (you run this — it needs YOUR OASIS-3 / XNAT credentials)
---------------------------------------------------------------------------
OASIS-3 is controlled-access; CRYCHIC cannot download it for you. With access set up
(https://www.oasis-brains.org → XNAT central.xnat.org), pull the subject's FLAIR
session, e.g. with the XNAT downloader or ``curl`` against the REST API, then point
``--flair`` at the resulting ``.nii.gz``.

Usage
-----
  python scripts/coreg_flair.py --subject OAS30073 --flair /path/to/sub-OAS30073_FLAIR.nii.gz
  python scripts/coreg_flair.py --subject OAS30073 --flair ... --no-bet   # FLAIR already brain-only

Requires FSL on PATH (flirt, bet, convert_xfm). ANTs (antsRegistration) is an
equally valid rigid alternative if you prefer — not wired here.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PREPRO = _REPO_ROOT / "data" / "mri_prepro"

# FSL writes .nii.gz when this is set; keep it explicit so output names are predictable.
_ENV = {**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"}


def _need(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        sys.exit(f"error: '{tool}' not found on PATH — install FSL or add it to PATH.")
    return path


def _run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, env=_ENV)


def _one(pattern: str) -> Path | None:
    hits = sorted(_PREPRO.glob(pattern))
    return hits[0] if hits else None


def coreg(subject: str, flair: Path, *, bet: bool = True) -> Path:
    """Align ``flair`` to the subject's MNI T1 grid; return the written FLAIR path."""
    t1_mni = _one(f"{subject}_*_stripped_MNI.nii.gz")
    if t1_mni is None:
        sys.exit(f"error: no stripped_MNI T1 for {subject} under {_PREPRO} — "
                 "run the T1 preprocessing first.")
    prefix = t1_mni.name.replace("_stripped_MNI.nii.gz", "")  # e.g. OAS30073_MR_d3670
    t1_native = _PREPRO / f"{prefix}_stripped.nii.gz"
    t1_native = t1_native if t1_native.exists() else None
    to_mni = _PREPRO / f"{prefix}_to_mni.mat"
    to_mni = to_mni if to_mni.exists() else None

    _need("flirt"); _need("convert_xfm")
    work = _PREPRO / f"{prefix}_flairtmp"
    work.mkdir(exist_ok=True)
    out = _PREPRO / f"{prefix}_FLAIR_MNI.nii.gz"

    flair_brain = flair
    if bet:
        _need("bet")
        flair_brain = work / "flair_brain.nii.gz"
        print("[1/4] skull-strip FLAIR (bet)")
        _run(["bet", str(flair), str(flair_brain), "-R", "-f", "0.4"])

    if t1_native is not None and to_mni is not None:
        print(f"[2/4] FLIRT FLAIR → native stripped T1 (6-dof rigid)\n      ref {t1_native.name}")
        flair2t1 = work / "flair2t1.mat"
        _run(["flirt", "-in", str(flair_brain), "-ref", str(t1_native),
              "-dof", "6", "-cost", "corratio", "-omat", str(flair2t1),
              "-out", str(work / "flair_in_t1.nii.gz")])
        print("[3/4] concat FLAIR→T1 with T1→MNI")
        flair2mni = work / "flair2mni.mat"
        _run(["convert_xfm", "-omat", str(flair2mni), "-concat", str(to_mni), str(flair2t1)])
        print(f"[4/4] resample FLAIR onto MNI grid\n      ref {t1_mni.name}")
        _run(["flirt", "-in", str(flair_brain), "-ref", str(t1_mni), "-applyxfm",
              "-init", str(flair2mni), "-out", str(out)])
    else:
        print("note: native stripped T1 / _to_mni.mat missing — direct FLAIR → stripped_MNI.")
        print(f"[2/2] FLIRT FLAIR → stripped_MNI T1 (6-dof rigid)\n      ref {t1_mni.name}")
        _run(["flirt", "-in", str(flair_brain), "-ref", str(t1_mni), "-dof", "6",
              "-cost", "corratio", "-out", str(out),
              "-omat", str(work / "flair2mni.mat")])

    print(f"\n✓ wrote {out}")
    print("  find_flair() will now discover it; re-run the case to exercise the VD axis.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Coregister FLAIR onto the MNI T1 grid.")
    ap.add_argument("--subject", required=True, help="OASIS subject id, e.g. OAS30073")
    ap.add_argument("--flair", required=True, type=Path, help="raw/native FLAIR .nii.gz")
    ap.add_argument("--no-bet", action="store_true",
                    help="skip skull-strip (FLAIR is already brain-only)")
    args = ap.parse_args()

    if not args.flair.exists():
        sys.exit(f"error: FLAIR not found: {args.flair}")
    coreg(args.subject, args.flair, bet=not args.no_bet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
