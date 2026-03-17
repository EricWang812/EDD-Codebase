#!/bin/bash
# run_translator.sh — launch the WhisPlay EN<->ZH translator on Raspberry Pi
# Usage:
#   sudo bash run_translator.sh
set -euo pipefail

echo "=== Sound cards (aplay -l) ==="
aplay -l 2>/dev/null || true
echo ""

# ── Find wm8960 card index ────────────────────────────────────────────────────
# Find the sound card index for wm8960soundcard
card_index=$(awk '/wm8960soundcard/ {print $1}' /proc/asound/cards | head -n1)
# Default to 1 if not found
if [ -z "$card_index" ]; then
  card_index=1
fi
echo "Using sound card index: $card_index"

AUDIODEV=hw:$card_index,0 python test.py $@

# ── Export audio env vars for Python ─────────────────────────────────────────
export WM8960_CARD_INDEX="$card_index"
export WM8960_CARD_NAME="wm8960soundcard"
export WM8960_HW="hw:${WM8960_CARD_NAME},0"
export AUDIODEV="hw:${card_index},0"

# ── Write a temporary ALSA config so arecord/aplay CLI tools use wm8960 ──────
ASOUNDRC_TMP=$(mktemp /tmp/.asoundrc.XXXXXX)
cleanup() { rm -f "$ASOUNDRC_TMP"; }
trap cleanup EXIT INT TERM

cat > "$ASOUNDRC_TMP" <<EOF
pcm.!default {
    type hw
    card $card_index
    device 0
}
ctl.!default {
    type hw
    card $card_index
}
EOF
export ALSA_CONFIG_PATH="$ASOUNDRC_TMP"

# ── Thread limits — important on Pi Zero 2W ──────────────────────────────────
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export ORT_NUM_THREADS=2
export ONNXRUNTIME_NUM_THREADS=2

export PYTHONUNBUFFERED=1

echo "AUDIODEV=$AUDIODEV"
echo "WM8960_CARD_INDEX=$WM8960_CARD_INDEX"
echo "WM8960_CARD_NAME=$WM8960_CARD_NAME"
echo "ALSA_CONFIG_PATH=$ALSA_CONFIG_PATH"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo ""


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PY="$SCRIPT_DIR/main.py"

if [ ! -f "$MAIN_PY" ]; then
  echo "[ERROR] main.py not found at $MAIN_PY"
  exit 1
fi

echo "Starting translator ..."
echo ""
exec python3 "$MAIN_PY" "$@"