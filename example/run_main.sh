#!/bin/bash
# run_translator.sh — launch the WhisPlay translator on Pi Zero 2W
# Usage: bash run_translator.sh

set -e

echo "=== Sound cards (aplay -l) ==="
aplay -l 2>/dev/null || true
echo ""

# ── Find wm8960 card index ────────────────────────────────────────────────────
# /proc/asound/cards format:  " 0 [wm8960soundcard]: ..."
card_index=$(awk '/wm8960soundcard/ {gsub(/[^0-9]/,"",$1); print $1; exit}' \
             /proc/asound/cards 2>/dev/null)

# Fallback via aplay -l
if [ -z "$card_index" ]; then
  card_index=$(aplay -l 2>/dev/null \
               | awk '/wm8960/ {
                   for (i=1; i<=NF; i++) {
                     if ($i == "card") {
                       gsub(/[^0-9]/, "", $(i+1))
                       print $(i+1)
                       exit
                     }
                   }
                 }')
fi

# Final fallback
if [ -z "$card_index" ]; then
  echo "[WARN] Could not detect wm8960 card — defaulting to card 1"
  card_index=1
fi

echo "Using wm8960 sound card index: $card_index"

# ── ALSA config: write a temp .asoundrc so ALSA libs pick the right card ─────
ASOUNDRC_TMP=$(mktemp /tmp/.asoundrc.XXXXXX)
cat > "$ASOUNDRC_TMP" <<EOF
pcm.!default {
    type hw
    card $card_index
}
ctl.!default {
    type hw
    card $card_index
}
EOF
export ALSA_CONFIG_PATH="$ASOUNDRC_TMP"

# Legacy env var
export AUDIODEV="hw:$card_index,0"

# ── Thread limits — critical for Pi Zero 2W ───────────────────────────────────
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export ORT_NUM_THREADS=2
export ONNXRUNTIME_NUM_THREADS=2

echo "ALSA_CONFIG_PATH=$ALSA_CONFIG_PATH"
echo "AUDIODEV=$AUDIODEV"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  ORT_NUM_THREADS=$ORT_NUM_THREADS"
echo ""
echo "Starting translator (startup takes 60-120s on Pi Zero 2W — please wait) ..."
echo ""

python main.py "$@"

# Cleanup temp file
rm -f "$ASOUNDRC_TMP"