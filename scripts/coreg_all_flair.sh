#!/usr/bin/env bash
#
# coreg_all_flair.sh — coregister every downloaded OASIS-3 FLAIR onto its MNI T1 grid.
#
# Run this AFTER download_oasis_scans.sh has pulled the FLAIR sessions listed in
# data/crychic_oasis_flair_sessions.csv. For each experiment id (e.g. OAS30073_MR_d3670)
# it locates the FLAIR .nii.gz the downloader wrote and feeds it to coreg_flair.py with
# the bare subject id (OAS30073), which is what coreg_flair.py / find_flair() expect.
#
# Usage:
#   ./scripts/coreg_all_flair.sh [download_dir] [sessions_csv] [extra coreg_flair.py args...]
#
# Defaults:
#   download_dir = data/oasis_flair      (matches the README's download command)
#   sessions_csv = data/crychic_oasis_flair_sessions.csv
#
# Examples:
#   ./scripts/coreg_all_flair.sh
#   ./scripts/coreg_all_flair.sh data/oasis_flair data/crychic_oasis_flair_sessions.csv --no-bet
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DOWNLOAD_DIR="${1:-data/oasis_flair}"
SESSIONS_CSV="${2:-data/crychic_oasis_flair_sessions.csv}"
# Any remaining args (after the first two) pass straight through to coreg_flair.py.
EXTRA_ARGS=()
if [ "$#" -gt 2 ]; then shift 2; EXTRA_ARGS=("$@"); fi

PY="${PYTHON:-python}"

if [ ! -f "$SESSIONS_CSV" ]; then
    echo "error: sessions csv not found: $SESSIONS_CSV" >&2
    exit 1
fi
if [ ! -d "$DOWNLOAD_DIR" ]; then
    echo "error: download dir not found: $DOWNLOAD_DIR" >&2
    echo "       run download_oasis_scans.sh first (see scripts/oasis_download/)." >&2
    exit 1
fi

ok=0; skipped=0; failed=0
declare -a SKIPPED=() FAILED=()

# Skip the header row, read one experiment id per line (tolerate trailing CR / extra cols).
while IFS=, read -r EXPERIMENT_ID _rest; do
    EXPERIMENT_ID="${EXPERIMENT_ID%$'\r'}"
    [ -z "$EXPERIMENT_ID" ] && continue
    SUBJECT_ID="${EXPERIMENT_ID%%_*}"   # OAS30073 from OAS30073_MR_d3670

    sess_dir="$DOWNLOAD_DIR/$EXPERIMENT_ID"
    if [ ! -d "$sess_dir" ]; then
        echo "— $EXPERIMENT_ID: no download folder ($sess_dir) — skipping."
        SKIPPED+=("$EXPERIMENT_ID (not downloaded)"); skipped=$((skipped+1)); continue
    fi

    # Prefer a *FLAIR*.nii.gz; fall back to any .nii.gz under the session folder.
    flair=""
    while IFS= read -r f; do flair="$f"; break; done < <(find "$sess_dir" -type f -iname '*flair*.nii.gz' 2>/dev/null | sort)
    if [ -z "$flair" ]; then
        while IFS= read -r f; do flair="$f"; break; done < <(find "$sess_dir" -type f -iname '*.nii.gz' 2>/dev/null | sort)
    fi
    if [ -z "$flair" ]; then
        echo "— $EXPERIMENT_ID: no .nii.gz found under $sess_dir — skipping."
        SKIPPED+=("$EXPERIMENT_ID (no nii.gz)"); skipped=$((skipped+1)); continue
    fi

    echo ""
    echo "=== $SUBJECT_ID  ($EXPERIMENT_ID) ==="
    echo "    flair: $flair"
    if "$PY" scripts/coreg_flair.py --subject "$SUBJECT_ID" --flair "$flair" "${EXTRA_ARGS[@]}"; then
        ok=$((ok+1))
    else
        echo "!! coreg failed for $SUBJECT_ID" >&2
        FAILED+=("$EXPERIMENT_ID"); failed=$((failed+1))
    fi
done < <(tail -n +2 "$SESSIONS_CSV")

echo ""
echo "================ summary ================"
echo "coregistered : $ok"
echo "skipped      : $skipped"
for s in "${SKIPPED[@]:-}"; do [ -n "$s" ] && echo "   - $s"; done
echo "failed       : $failed"
for s in "${FAILED[@]:-}"; do [ -n "$s" ] && echo "   - $s"; done

# Non-zero exit if anything failed, so it composes in a larger pipeline.
[ "$failed" -eq 0 ]
