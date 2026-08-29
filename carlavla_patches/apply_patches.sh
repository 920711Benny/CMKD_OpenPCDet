#!/usr/bin/env bash
# Apply the CarlaVLA patches into a simlingo_training tree.
#
#   ./apply_patches.sh /path/to/simlingo            (dry run: shows what it would do)
#   ./apply_patches.sh /path/to/simlingo --write    (actually copies)
#
# Every replaced file is backed up as <name>.bak.<timestamp> first. Nothing is
# overwritten without a backup, and a second run will not clobber the first
# backup.
set -euo pipefail

REPO="${1:-}"
WRITE="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [ -z "$REPO" ]; then
  sed -n '2,12p' "$0"; exit 1
fi
if [ ! -d "$REPO/simlingo_training" ]; then
  echo "ERROR: $REPO/simlingo_training not found -- is that the simlingo repo root?" >&2
  exit 1
fi

ENC="$REPO/simlingo_training/models/encoder"
MODELS="$REPO/simlingo_training/models"

declare -a PAIRS=(
  "$HERE/flash_mha.py|$ENC/flash_mha.py"
  "$HERE/dual_vision_model.py|$ENC/dual_vision_model.py"
  "$HERE/diffusion_decoder.py|$MODELS/diffusion_decoder.py"
)

echo "target repo : $REPO"
echo "mode        : $([ "$WRITE" = "--write" ] && echo WRITE || echo "DRY RUN (pass --write to apply)")"
echo

for pair in "${PAIRS[@]}"; do
  src="${pair%%|*}"; dst="${pair##*|}"
  if [ ! -f "$src" ]; then echo "ERROR: missing $src" >&2; exit 1; fi
  if [ -f "$dst" ]; then
    echo "  replace  $dst"
    echo "    backup $dst.bak.$STAMP"
  else
    echo "  create   $dst"
  fi
  if [ "$WRITE" = "--write" ]; then
    mkdir -p "$(dirname "$dst")"
    [ -f "$dst" ] && cp -p "$dst" "$dst.bak.$STAMP"
    cp "$src" "$dst"
  fi
done

echo
if [ "$WRITE" = "--write" ]; then
  echo "Applied. To revert:  for f in \$(find $REPO -name '*.bak.$STAMP'); do mv \"\$f\" \"\${f%.bak.$STAMP}\"; done"
  echo
  echo "diffusion_decoder.py now defaults to prediction_type='v'. That changes the"
  echo "training target, so an existing checkpoint will NOT transfer -- pass"
  echo "prediction_type='epsilon' to keep the old behaviour."
  echo "Set coord_min_max from your data: see fit_coord_range in the same file."
else
  echo "Nothing written. Re-run with --write to apply."
fi
