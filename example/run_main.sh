#!/bin/bash
# run_translator.sh — launch the WhisPlay EN<->ZH translator on Raspberry Pi
# Usage:
#   sudo bash run_translator.sh
set -euo pipefail

echo "=== Sound cards (aplay -l) ==="
aplay -l 2>/dev/null || true
echo ""

# ── Find wm8960 card index ────────────────────────────────────────────────────
find_wm8960_card_index() {
  local idx=""

  # Primary: parse /proc/asound/cards
  # Format: " 0 [wm8960soundcard]: ..."
  if [ -r /proc/asound/cards ]; then
    idx=$(awk '
      /wm8960soundcard|wm8960/ {
        gsub(/[^0-9]/, "", $1)
        if ($1 != "") { print $1; exit }
      }
    ' /proc/asound/cards 2>/dev/null || true)
  fi

  # Fallback: parse aplay -l
  if [ -z "$idx" ]; then
    idx=$(aplay -l 2>/dev/null | awk '
      /wm8960/ {
        for (i = 1; i <= NF; i++) {
          if ($i == "card") {
            x = $(i+1)
            gsub(/[^0-9]/, "", x)
            if (x != "") { print x; exit }
          }
        }
      }
    ' || true)
  fi

  if [ -n "$idx" ]; then
    printf '%s\n' "$idx"
    return 0
  fi
  return 1
}

if card_index=$(find_wm8960_card_index); then
  echo "Found wm8960 at card index: $card_index"
else
  echo "[WARN] Could not detect wm8960 card — defaulting to card 1"
  card_index=1
fi

# ── Export audio env vars for Python ─────────────────────────────────────────
# WM8960_CARD_INDEX is used by main.py's _find_sd_device() to locate
# the correct sounddevice device via the "hw:N" pattern — same method
# the test .sh uses with AUDIODEV=hw:N,0.
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

# ── Sanity checks ─────────────────────────────────────────────────────────────
if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found in PATH"
  exit 1
fi
if ! command -v arecord >/dev/null 2>&1; then
  echo "[ERROR] arecord not found — install with: sudo apt install -y alsa-utils"
  exit 1
fi
if ! command -v aplay >/dev/null 2>&1; then
  echo "[ERROR] aplay not found — install with: sudo apt install -y alsa-utils"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PY="$SCRIPT_DIR/main.py"

if [ ! -f "$MAIN_PY" ]; then
  echo "[ERROR] main.py not found at $MAIN_PY"
  exit 1
fi

echo "Starting translator ..."
echo ""
exec python3 "$MAIN_PY" "$@"