#!/usr/bin/env bash
# Demo for the ADRD nmed2024 tool. Runs two calls on the bundled example case:
#   1. clinical-only  -> predictions + Feature x label attribution heatmap
#   2. clinical + MRI  -> same, with an extra "MRI (img)" row (needs an embedding)
#
# Usage:  ./demo.sh [PYTHON] [OUT_DIR] [MRI]
#   PYTHON   python interpreter (default: python3)
#   OUT_DIR  where the heatmap is written (default: ./demo_out)
#   MRI      optional .npy embedding or preprocessed .nii.gz
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${1:-python3}"
OUT_DIR="${2:-$HERE/demo_out}"
MRI="${3:-}"
EXAMPLE="$HERE/examples/example_clinical.json"

cd "$HERE"
echo "=============================================================="
echo " ADRD nmed2024 tool demo  |  python: $PY  |  out: $OUT_DIR"
echo "=============================================================="

echo
echo "########## [1/2] clinical-only ##########"
"$PY" adrd_tool.py --clinical "$EXAMPLE" --out "$OUT_DIR"

echo
echo "########## [2/2] clinical + MRI ##########"
if [[ -n "$MRI" && -f "$MRI" ]]; then
    "$PY" adrd_tool.py --clinical "$EXAMPLE" --mri "$MRI" --out "$OUT_DIR"
else
    echo "(skipped — pass an .npy embedding or .nii.gz as the 3rd arg to enable)"
fi

echo
echo "heatmap written to: $OUT_DIR"
ls -la "$OUT_DIR" 2>/dev/null || true
