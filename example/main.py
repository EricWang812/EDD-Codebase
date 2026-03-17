"""
WhisPlay HAT Translator — EN <-> ZH
Vosk STT  +  CTranslate2 OPUS MT  +  Piper TTS

Hardware: WhisPlay HAT (Raspberry Pi / Radxa)
  - Button long press  (≥1s) = start with English speaker first (EN→ZH)
  - Button short press (<1s) = start with Chinese speaker first (ZH→EN)
  - Press again while recording = stop → translate → speak → swap sides
  - 10s inactivity timeout = return to IDLE, requires fresh button press

Install deps:
    pip install vosk ctranslate2 sentencepiece sounddevice numpy piper-tts pillow
    sudo apt install espeak-ng libportaudio2 portaudio19-dev alsa-utils
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import time
import threading
import wave
from dataclasses import dataclass
from enum import Enum
from time import sleep
from typing import List, Optional

import numpy as np
import sounddevice as sd
import vosk
import ctranslate2
import sentencepiece as spm

try:
    from piper.voice import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("[WARN] piper-tts not installed — run: pip install piper-tts")

# Resolve Driver path relative to THIS file so it works regardless of
# what directory you launch the script from.
_DRIVER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Driver")
)
sys.path.insert(0, _DRIVER_DIR)
from WhisPlay import WhisPlayBoard

# ─────────────────────────────────────────────────────────────────────────────
# State Machine
# ─────────────────────────────────────────────────────────────────────────────
class State(Enum):
    IDLE       = 0
    LISTENING  = 1
    PROCESSING = 2

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DATA_DIR   = os.path.join(HERE, "data")
VOICES_DIR = os.path.join(HERE, "voices")
IMGS_DIR   = os.path.join(HERE, "imgs")
os.makedirs(DATA_DIR, exist_ok=True)

REC_FILE      = os.path.join(DATA_DIR, "recorded_voice.wav")
TTS_OUT_FILE  = os.path.join(DATA_DIR, "tts_out.wav")
VOSK_EN_DIR   = os.path.join(MODELS_DIR, "vosk-en")
VOSK_ZH_DIR   = os.path.join(MODELS_DIR, "vosk-cn")
CT2_EN_ZH_DIR = os.path.join(MODELS_DIR, "opus-en-zh-ct2")
CT2_ZH_EN_DIR = os.path.join(MODELS_DIR, "opus-zh-en-ct2")
VOICE_EN      = os.path.join(VOICES_DIR, "en_US-lessac-low.onnx")
VOICE_ZH      = os.path.join(VOICES_DIR, "zh_CN-huayan-x_low.onnx")

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
CT2_COMPUTE_TYPE       = "int8"
CT2_BEAM_SIZE          = 2
CT2_MAX_DECODING_LEN   = 256
CT2_NO_REPEAT_NGRAM    = 3
CT2_REPETITION_PENALTY = 1.15

MIC_SAMPLE_RATE  = 16_000
MIC_CHANNELS     = 1
MIC_BLOCK_SEC    = 0.25

HOLD_DURATION  = 1.0   # seconds — long press threshold
CONVO_TIMEOUT  = 10.0  # seconds — inactivity → back to IDLE

ENABLE_TTS = True

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MTModel:
    translator: ctranslate2.Translator
    sp_src:     spm.SentencePieceProcessor
    sp_tgt:     spm.SentencePieceProcessor

@dataclass
class AppState:
    vosk_en:  vosk.Model
    vosk_zh:  vosk.Model
    en_zh:    MTModel
    zh_en:    MTModel
    piper_en: object
    piper_zh: object

# ─────────────────────────────────────────────────────────────────────────────
# Audio device detection (input only — aplay handles output)
# ─────────────────────────────────────────────────────────────────────────────
def _find_sd_input_device() -> Optional[int]:
    """
    Find the wm8960 sounddevice index for input (microphone).
    Strategy:
      1. Match 'hw:<WM8960_CARD_INDEX>' in device name  (most reliable)
      2. Match WM8960_CARD_NAME substring in device name
      3. Match 'wm8960' anywhere in device name
      4. Return None (sounddevice default)
    """
    card_idx  = os.environ.get("WM8960_CARD_INDEX", "").strip()
    card_name = os.environ.get("WM8960_CARD_NAME", "wm8960soundcard").lower()

    try:
        devices = sd.query_devices()

        # Strategy 1 — hw:N prefix
        if card_idx.isdigit():
            hw = f"hw:{card_idx}"
            for i, d in enumerate(devices):
                if hw in d["name"] and d["max_input_channels"] > 0:
                    print(f"[Audio] input device {i}: {d['name']!r}")
                    return i

        # Strategy 2 — card name substring
        for i, d in enumerate(devices):
            if card_name in d["name"].lower() and d["max_input_channels"] > 0:
                print(f"[Audio] input device {i}: {d['name']!r}")
                return i

        # Strategy 3 — wm8960 anywhere
        for i, d in enumerate(devices):
            if "wm8960" in d["name"].lower() and d["max_input_channels"] > 0:
                print(f"[Audio] input device {i}: {d['name']!r}")
                return i

    except Exception as e:
        print(f"[Audio] Device scan error: {e}")

    print("[Audio] WARNING: wm8960 input not found — using sounddevice default")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_image(board, filepath: str):
    """Load an image as RGB565 for the WhisPlay LCD with aspect-ratio crop."""
    try:
        from PIL import Image
        img = Image.open(filepath).convert("RGB")
        ow, oh = img.size
        sw, sh = board.LCD_WIDTH, board.LCD_HEIGHT

        if (ow / oh) > (sw / sh):
            nh = sh
            nw = int(nh * ow / oh)
            img = img.resize((nw, nh))
            ox = (nw - sw) // 2
            img = img.crop((ox, 0, ox + sw, sh))
        else:
            nw = sw
            nh = int(nw * oh / ow)
            img = img.resize((nw, nh))
            oy = (nh - sh) // 2
            img = img.crop((0, oy, sw, oy + sh))

        data = []
        for y in range(sh):
            for x in range(sw):
                r, g, b = img.getpixel((x, y))
                rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
                data.extend([(rgb565 >> 8) & 0xFF, rgb565 & 0xFF])
        return data
    except Exception as e:
        print(f"  [Display] Could not load image {filepath}: {e}")
        return None


def update_display(board, state: State, images: dict):
    key = state.name.lower()  # "idle" / "listening" / "processing"
    img = images.get(key)
    if img:
        board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def _load_ct2(model_dir: str) -> MTModel:
    for f in ("model.bin", "source.spm", "target.spm"):
        p = os.path.join(model_dir, f)
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    translator = ctranslate2.Translator(model_dir, compute_type=CT2_COMPUTE_TYPE)
    sp_src = spm.SentencePieceProcessor()
    sp_src.load(os.path.join(model_dir, "source.spm"))
    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.load(os.path.join(model_dir, "target.spm"))
    return MTModel(translator, sp_src, sp_tgt)


def _load_piper_voice(onnx_path: str):
    if not PIPER_AVAILABLE:
        return None
    json_path = onnx_path + ".json"
    if not os.path.isfile(onnx_path):
        print(f"  [Piper] voice not found: {onnx_path}")
        return None
    try:
        return PiperVoice.load(
            onnx_path,
            config_path=json_path if os.path.isfile(json_path) else None,
            use_cuda=False,
        )
    except Exception as e:
        print(f"  [Piper] load error: {e}")
        return None


def init_app() -> AppState:
    print("Loading Vosk EN ...")
    vosk_en = vosk.Model(VOSK_EN_DIR)
    print("Loading Vosk ZH ...")
    vosk_zh = vosk.Model(VOSK_ZH_DIR)
    print("Loading EN->ZH model ...")
    en_zh = _load_ct2(CT2_EN_ZH_DIR)
    print("Loading ZH->EN model ...")
    zh_en = _load_ct2(CT2_ZH_EN_DIR)

    piper_en = piper_zh = None
    if PIPER_AVAILABLE:
        print("Loading Piper EN voice ...")
        piper_en = _load_piper_voice(VOICE_EN)
        print("Loading Piper ZH voice ...")
        piper_zh = _load_piper_voice(VOICE_ZH)
        loaded = [n for n, v in [("EN", piper_en), ("ZH", piper_zh)] if v]
        print(f"Piper voices loaded: {loaded if loaded else 'none'}")

    print("\nReady.\n")
    return AppState(vosk_en, vosk_zh, en_zh, zh_en, piper_en, piper_zh)

# ─────────────────────────────────────────────────────────────────────────────
# Audio pre-processing
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.abs(audio.astype(np.float32)).max()
    if peak < 1:
        return audio
    scale = (32767 * 0.9) / peak
    if scale < 1.0:
        return audio
    return np.clip(audio.astype(np.float32) * scale, -32768, 32767).astype(np.int16)

# ─────────────────────────────────────────────────────────────────────────────
# Recording
# ─────────────────────────────────────────────────────────────────────────────
def record_until_button(stop_event, in_device=None) -> str:
    MAX_RECORD_SEC = 30
    block_size = int(MIC_SAMPLE_RATE * MIC_BLOCK_SEC)
    max_blocks = int(MAX_RECORD_SEC / MIC_BLOCK_SEC)
    frames = []

    print("  [MIC] Recording — press button to stop ...")
    with sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=MIC_CHANNELS,
                        dtype="int16", blocksize=block_size,
                        device=in_device) as stream:
        for _ in range(max_blocks):
            if stop_event.is_set():
                break
            block, _ = stream.read(block_size)
            frames.append(block.copy())

    print("  [MIC] Recording stopped.")
    audio = np.concatenate(frames, axis=0)
    with wave.open(REC_FILE, "wb") as wf:
        wf.setnchannels(MIC_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(MIC_SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return REC_FILE

# ─────────────────────────────────────────────────────────────────────────────
# STT
# ─────────────────────────────────────────────────────────────────────────────
_FILLER_RE = re.compile(r'\b(uh+|um+|hmm+|huh|mm+|ah+|er+)\b', re.IGNORECASE)

def _strip_fillers(text: str) -> str:
    return ' '.join(_FILLER_RE.sub('', text).split())


def _transcribe_wav(model: vosk.Model, wav_path: str) -> str:
    with wave.open(wav_path, "rb") as wf:
        raw  = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
        nch  = wf.getnchannels()
        sw   = wf.getsampwidth()

    audio = _normalize_audio(np.frombuffer(raw, dtype=np.int16))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf_out:
        wf_out.setnchannels(nch)
        wf_out.setsampwidth(sw)
        wf_out.setframerate(rate)
        wf_out.writeframes(audio.tobytes())
    buf.seek(0)

    rec = vosk.KaldiRecognizer(model, rate)
    rec.SetWords(True)
    parts        = []
    last_partial = ""

    while True:
        data = buf.read(8000)
        if not data:
            break
        if rec.AcceptWaveform(data):
            seg = _strip_fillers(json.loads(rec.Result()).get("text", ""))
            if seg:
                parts.append(seg)
            last_partial = ""
        else:
            last_partial = json.loads(rec.PartialResult()).get("partial", "")

    seg = _strip_fillers(json.loads(rec.FinalResult()).get("text", ""))
    if seg:
        parts.append(seg)
    elif last_partial:
        seg = _strip_fillers(last_partial)
        if seg:
            parts.append(seg)

    return " ".join(parts).strip()


def _dedup_stt(text: str) -> str:
    words = text.split()
    n     = len(words)
    if n == 0:
        return text
    half = n // 2
    if half > 0 and words[:half] == words[half:half * 2]:
        return _dedup_stt(" ".join(words[half:]))
    for size in range(half, 0, -1):
        if words[:size] == words[size:size * 2]:
            return " ".join(words[size:])
    deduped = [words[0]]
    for w in words[1:]:
        if w != deduped[-1]:
            deduped.append(w)
    return " ".join(deduped)

# ─────────────────────────────────────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────────────────────────────────────
def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[。！？；.!?;])\s*', text)
    return [p.strip() for p in parts if p.strip()]


def _clean(text: str) -> str:
    text = re.sub(r'([\u4e00-\u9fff\u3000-\u303f]{2,6})\1{2,}', r'\1', text)
    return " ".join(text.split()).strip()


def translate(mt: MTModel, text: str) -> str:
    if not text.strip():
        return ""
    results = []
    for sent in (_split_sentences(text) or [text]):
        toks = mt.sp_src.encode(sent, out_type=str)
        if mt.sp_src.piece_to_id("</s>") != mt.sp_src.unk_id():
            toks = toks + ["</s>"]
        res = mt.translator.translate_batch(
            [toks],
            beam_size=CT2_BEAM_SIZE,
            max_decoding_length=CT2_MAX_DECODING_LEN,
            no_repeat_ngram_size=CT2_NO_REPEAT_NGRAM,
            repetition_penalty=CT2_REPETITION_PENALTY,
        )
        out = [t for t in res[0].hypotheses[0] if t not in {"</s>", "<s>", "<pad>"}]
        results.append(mt.sp_tgt.decode(out).strip())
    return _clean(" ".join(results))

# ─────────────────────────────────────────────────────────────────────────────
# TTS
# Piper synthesizes to a WAV file, then aplay plays it back directly on the
# wm8960 card — no pygame, no SDL, no device guessing.
# aplay -D hw:<card>,0 is the same device the .sh already verified works.
# ─────────────────────────────────────────────────────────────────────────────
def speak(app: AppState, text: str, lang: str, hw_device: str):
    if not ENABLE_TTS or not text:
        return

    voice = app.piper_zh if lang == "zh" else app.piper_en
    if not voice:
        print(f"  [TTS] No voice loaded for lang={lang}")
        return

    try:
        # Synthesize to WAV
        rate = getattr(getattr(voice, "config", None), "sample_rate", 22050)
        with wave.open(TTS_OUT_FILE, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            voice.synthesize(text, wf)

        # Play with aplay pointing directly at the wm8960 hw device
        subprocess.run(
            ["aplay", "-D", hw_device, TTS_OUT_FILE],
            check=True
        )

    except subprocess.CalledProcessError as e:
        print(f"  [TTS] aplay error: {e}")
    except Exception as e:
        print(f"  [TTS] Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── STEP 1: Board + idle image ────────────────────────────────────────────
    board = WhisPlayBoard()
    board.set_backlight(50)

    images = {
        "idle":       _load_image(board, os.path.join(IMGS_DIR, "passive.jpg")),
        "listening":  _load_image(board, os.path.join(IMGS_DIR, "listening.jpg")),
        "processing": _load_image(board, os.path.join(IMGS_DIR, "talking.jpg")),
    }

    if images["idle"]:
        board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, images["idle"])
        print("[Display] Idle image shown.")
    else:
        print("[Display] WARNING: idle image missing or failed to load.")

    # ── STEP 2: Resolve hw device string from env (set by .sh) ───────────────
    card_index = os.environ.get("WM8960_CARD_INDEX", "1").strip()
    hw_device  = f"plughw:{card_index},0"
    print(f"[Audio] output hw device: {hw_device}")

    # ── STEP 3: Detect mic input device ──────────────────────────────────────
    in_device = _find_sd_input_device()

    # ── STEP 4: Set speaker volume ────────────────────────────────────────────
    card_name = os.environ.get("WM8960_CARD_NAME", "wm8960soundcard")
    try:
        subprocess.run(
            ["amixer", "-D", f"hw:{card_name}", "sset", "Speaker", "121"],
            check=True, capture_output=True, text=True
        )
        print("[Audio] Speaker volume set to 121")
    except Exception as e:
        print(f"[Audio] Could not set speaker volume: {e}")

    # ── STEP 5: Load models ───────────────────────────────────────────────────
    app = init_app()

    # ── Shared mutable state ──────────────────────────────────────────────────
    state          = [State.IDLE]
    eng_to_cn      = [True]
    last_activity  = [0.0]
    press_time     = [0.0]
    stop_recording = threading.Event()
    aborted        = [False]

    def set_state(new_state: State):
        state[0] = new_state
        label = {
            State.IDLE:       "IDLE       — waiting for button",
            State.LISTENING:  "LISTENING  — recording ...",
            State.PROCESSING: "PROCESSING — translating & speaking ...",
        }[new_state]
        print(f"\n[STATE] {label}")
        update_display(board, new_state, images)

    # ── Button callbacks ──────────────────────────────────────────────────────
    def on_press():
        if state[0] == State.IDLE:
            press_time[0] = time.time()
        elif state[0] == State.LISTENING:
            print("[BTN] Press → stopping recording")
            stop_recording.set()

    def on_release():
        if state[0] == State.IDLE:
            duration = time.time() - press_time[0]
            if duration >= HOLD_DURATION:
                eng_to_cn[0] = True
                print(f"[BTN] Long press ({duration:.2f}s) → English first (EN→ZH)")
            else:
                eng_to_cn[0] = False
                print(f"[BTN] Short press ({duration:.2f}s) → Chinese first (ZH→EN)")

            last_activity[0] = time.time()
            stop_recording.clear()
            set_state(State.LISTENING)
            threading.Thread(target=_recording_thread, daemon=True).start()

    # ── Recording + processing thread ────────────────────────────────────────
    def _recording_thread():
        record_until_button(stop_recording, in_device=in_device)

        if aborted[0]:
            aborted[0] = False
            return

        set_state(State.PROCESSING)

        is_en_first = eng_to_cn[0]
        direction   = "EN→ZH" if is_en_first else "ZH→EN"
        vosk_model  = app.vosk_en if is_en_first else app.vosk_zh
        mt          = app.en_zh   if is_en_first else app.zh_en
        tgt_lang    = "zh"        if is_en_first else "en"

        raw  = _transcribe_wav(vosk_model, REC_FILE)
        text = _dedup_stt(raw)
        print(f"  Recognised: {text}" + (f"  (raw: {raw})" if raw != text else ""))

        if not text:
            print("  Nothing recognised — returning to LISTEN.")
            last_activity[0] = time.time()
            stop_recording.clear()
            set_state(State.LISTENING)
            threading.Thread(target=_recording_thread, daemon=True).start()
            return

        t0  = time.time()
        out = translate(mt, text)
        print(f"  [{direction}] {text}\n  →  {out}  [{time.time()-t0:.2f}s]")

        speak(app, out, tgt_lang, hw_device)

        eng_to_cn[0]     = not eng_to_cn[0]
        last_activity[0] = time.time()

        stop_recording.clear()
        set_state(State.LISTENING)
        threading.Thread(target=_recording_thread, daemon=True).start()

    # ── Register button callbacks ─────────────────────────────────────────────
    board.on_button_press(on_press)
    board.on_button_release(on_release)

    # ── Show idle state ───────────────────────────────────────────────────────
    set_state(State.IDLE)
    print("Button: LONG press  (hold ≥1s then release) = English speaker first (EN→ZH)")
    print("        SHORT press (tap and release)        = Chinese speaker first (ZH→EN)")

    # ── Main loop — inactivity watchdog ──────────────────────────────────────
    try:
        while True:
            if state[0] != State.IDLE:
                if time.time() - last_activity[0] > CONVO_TIMEOUT:
                    print("\n[TIMEOUT] 10s inactivity — returning to IDLE.")
                    aborted[0] = True
                    stop_recording.set()
                    sleep(0.5)
                    stop_recording.clear()
                    set_state(State.IDLE)
            sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting ...")
        stop_recording.set()
        board.cleanup()


if __name__ == "__main__":
    main()