"""
WhisPlay HAT Translator — EN <-> TH
Vosk STT  +  CTranslate2 HPLT MT  +  PyThaiTTS (Thai) / espeak-ng (English)

Hardware: WhisPlay HAT (Raspberry Pi / Radxa)
  - Button very long press (>=2.0s) = shutdown
  - Button long press  (>=0.4s) = start with English speaker first (EN->TH)
  - Button short press (<0.4s)  = start with Thai speaker first (TH->EN)
  - Press again while recording = stop -> translate -> speak -> swap sides
  - 10s inactivity timeout = return to IDLE, requires fresh button press

Install deps:
    pip install vosk ctranslate2 sentencepiece sounddevice numpy pythaitts pythainlp pillow
    sudo apt install espeak-ng libportaudio2 portaudio19-dev alsa-utils

Model setup:
    # Download and convert HPLT Marian models to CTranslate2 format
    pip install huggingface_hub
    python -c "
    from huggingface_hub import snapshot_download
    snapshot_download('HPLT/translate-th-en-v2.0-hplt', local_dir='models/hplt-th-en')
    snapshot_download('HPLT/translate-en-th-v2.0-hplt', local_dir='models/hplt-en-th')
    "
    ct2-opus-mt-converter \
        --model_path models/hplt-th-en/model.npz.best-chrf.npz \
        --vocab_path  models/hplt-th-en/model.th-en.spm \
        --output_dir  models/hplt-th-en-ct2 \
        --quantization int8
    ct2-opus-mt-converter \
        --model_path models/hplt-en-th/model.npz.best-chrf.npz \
        --vocab_path  models/hplt-en-th/model.en-th.spm \
        --output_dir  models/hplt-en-th-ct2 \
        --quantization int8

    # Download Vosk Thai model (vistec-AI/commonvoice-th release)
    mkdir -p models/vosk-th
    wget https://github.com/vistec-AI/commonvoice-th/releases/download/vosk-v1/model.zip
    unzip model.zip -d models/vosk-th
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

# PyThaiTTS — Thai TTS
try:
    from pythaitts import TTS as ThaiTTS
    PYTHAITTS_AVAILABLE = True
except ImportError:
    PYTHAITTS_AVAILABLE = False
    print("[WARN] pythaitts not installed — run: pip install pythaitts")

# PyThaiNLP — Thai word tokeniser (used before translation)
try:
    from pythainlp.tokenize import word_tokenize as th_word_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    print("[WARN] pythainlp not installed — run: pip install pythainlp")

_DRIVER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Driver")
)
sys.path.insert(0, _DRIVER_DIR)
from WhisPlay import WhisPlayBoard


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
class State(Enum):
    IDLE       = 0
    LISTENING  = 1
    PROCESSING = 2


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DATA_DIR   = os.path.join(HERE, "data")
IMGS_DIR   = os.path.join(HERE, "imgs")
os.makedirs(DATA_DIR, exist_ok=True)

REC_FILE     = os.path.join(DATA_DIR, "recorded_voice.wav")
TTS_OUT_FILE = os.path.join(DATA_DIR, "tts_out.wav")

# Vosk models
VOSK_EN_DIR = os.path.join(MODELS_DIR, "vosk-en")
VOSK_TH_DIR = os.path.join(MODELS_DIR, "vosk-th")

# CTranslate2 HPLT models (converted from Marian via ct2-opus-mt-converter)
CT2_EN_TH_DIR = os.path.join(MODELS_DIR, "hplt-en-th-ct2")
CT2_TH_EN_DIR = os.path.join(MODELS_DIR, "hplt-th-en-ct2")

# SPM vocab files (from the original Marian download)
SPM_EN_TH = os.path.join(MODELS_DIR, "hplt-en-th", "model.en-th.spm")
SPM_TH_EN = os.path.join(MODELS_DIR, "hplt-th-en", "model.th-en.spm")


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
CT2_COMPUTE_TYPE       = "int8"
CT2_BEAM_SIZE          = 4
CT2_MAX_DECODING_LEN   = 256
CT2_NO_REPEAT_NGRAM    = 3
CT2_REPETITION_PENALTY = 1.15

MIC_SAMPLE_RATE = 16_000
MIC_CHANNELS    = 1
MIC_BLOCK_SEC   = 0.25

HOLD_DURATION   = 0.4   # seconds — long press threshold
SHUTOFF_DURATION = 2.0  # seconds — very long press = shutdown
CONVO_TIMEOUT   = 10.0  # seconds — idle timeout during conversation

ENABLE_TTS = True

# PyThaiTTS model choice: "lunarlist_onnx" | "vachana" | "khanomtan"
THAI_TTS_MODEL  = "lunarlist_onnx"
# vachana speaker options: th_f_1, th_m_1, th_f_2, th_m_2
THAI_TTS_SPEAKER = "th_f_1"

# espeak-ng voice for English output
ESPEAK_EN_VOICE = "en-us"
ESPEAK_RATE     = 160   # words-per-minute

APLAY_DEVICE = "plughw:CARD=wm8960soundcard,DEV=0"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class MTModel:
    translator: ctranslate2.Translator
    sp:         spm.SentencePieceProcessor  # shared src+tgt sentencepiece vocab


@dataclass
class AppState:
    vosk_en:   vosk.Model
    vosk_th:   vosk.Model
    en_th:     MTModel
    th_en:     MTModel
    thai_tts:  object   # ThaiTTS instance or None


# ---------------------------------------------------------------------------
# Audio device detection
# ---------------------------------------------------------------------------
def _find_sd_input_device() -> Optional[int]:
    card_idx  = os.environ.get("WM8960_CARD_INDEX", "").strip()
    card_name = os.environ.get("WM8960_CARD_NAME", "wm8960soundcard").lower()
    try:
        devices = sd.query_devices()

        for i, d in enumerate(devices):
            if card_idx.isdigit() and f"hw:{card_idx}" in d["name"] and d["max_input_channels"] > 0:
                print(f"[Audio] input device {i}: {d['name']!r}")
                return i
        for i, d in enumerate(devices):
            if card_name in d["name"].lower() and d["max_input_channels"] > 0:
                print(f"[Audio] input device {i}: {d['name']!r}")
                return i
        for i, d in enumerate(devices):
            if "wm8960" in d["name"].lower() and d["max_input_channels"] > 0:
                print(f"[Audio] input device {i}: {d['name']!r}")
                return i
        for i, d in enumerate(devices):
            if card_idx.isdigit() and f":{card_idx}" in d["name"] and d["max_input_channels"] > 0:
                print(f"[Audio] input device {i}: {d['name']!r}")
                return i
        for i, d in enumerate(devices):
            if "simple-card" in d["name"].lower() and d["max_input_channels"] > 0:
                print(f"[Audio] input device {i}: {d['name']!r}")
                return i
    except Exception as e:
        print(f"[Audio] Device scan error: {e}")

    print("[Audio] WARNING: wm8960 input not found — using sounddevice default")
    return None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
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
    key = state.name.lower()
    img = images.get(key)
    if img:
        board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_ct2(model_dir: str, spm_path: str) -> MTModel:
    """Load a CTranslate2 Marian model with its shared SPM vocab file."""
    if not os.path.isfile(os.path.join(model_dir, "model.bin")):
        raise FileNotFoundError(f"CT2 model.bin missing in: {model_dir}")
    if not os.path.isfile(spm_path):
        raise FileNotFoundError(f"SPM vocab missing: {spm_path}")

    translator = ctranslate2.Translator(model_dir, compute_type=CT2_COMPUTE_TYPE)
    sp = spm.SentencePieceProcessor()
    sp.load(spm_path)
    return MTModel(translator, sp)


def _load_thai_tts() -> Optional[object]:
    """Load PyThaiTTS model. Returns None if unavailable."""
    if not PYTHAITTS_AVAILABLE:
        return None
    try:
        print(f"Loading PyThaiTTS ({THAI_TTS_MODEL}) ...")
        tts = ThaiTTS(pretrained=THAI_TTS_MODEL)
        print("PyThaiTTS loaded.")
        return tts
    except Exception as e:
        print(f"[WARN] PyThaiTTS failed to load: {e}")
        return None


def init_app() -> AppState:
    print("Loading Vosk EN ...")
    vosk_en = vosk.Model(VOSK_EN_DIR)
    print("Loading Vosk TH ...")
    vosk_th = vosk.Model(VOSK_TH_DIR)
    print("Loading EN->TH model ...")
    en_th = _load_ct2(CT2_EN_TH_DIR, SPM_EN_TH)
    print("Loading TH->EN model ...")
    th_en = _load_ct2(CT2_TH_EN_DIR, SPM_TH_EN)
    thai_tts = _load_thai_tts()

    print("\nReady.\n")
    return AppState(vosk_en, vosk_th, en_th, th_en, thai_tts)


# ---------------------------------------------------------------------------
# Audio pre-processing
# ---------------------------------------------------------------------------
def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.abs(audio.astype(np.float32)).max()
    if peak < 1:
        return audio
    scale = (32767 * 0.9) / peak
    if scale < 1.0:
        return audio
    return np.clip(audio.astype(np.float32) * scale, -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def record_until_button(stop_event, in_device=None) -> str:
    MAX_RECORD_SEC = 30
    block_size = int(MIC_SAMPLE_RATE * MIC_BLOCK_SEC)
    max_blocks = int(MAX_RECORD_SEC / MIC_BLOCK_SEC)
    frames = []

    with sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=MIC_CHANNELS,
                        dtype="int16", blocksize=block_size,
                        device=in_device) as stream:
        for _ in range(max_blocks):
            if stop_event.is_set():
                break
            block, _ = stream.read(block_size)
            frames.append(block.copy())

    audio = np.concatenate(frames, axis=0)
    with wave.open(REC_FILE, "wb") as wf:
        wf.setnchannels(MIC_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(MIC_SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return REC_FILE


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Thai word tokenisation (needed before feeding Thai text to MT model)
# ---------------------------------------------------------------------------
def _tokenize_thai_for_mt(text: str) -> str:
    """
    Insert spaces between Thai words so the SPM tokeniser sees clean input.
    Falls back to raw text if PyThaiNLP is unavailable.
    """
    if not PYTHAINLP_AVAILABLE:
        return text
    tokens = th_word_tokenize(text, engine="newmm")
    return " ".join(t for t in tokens if t.strip())


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[。！？；.!?;])\s*', text)
    return [p.strip() for p in parts if p.strip()]


def _clean(text: str) -> str:
    # Collapse repeated Thai substrings (MT artifact)
    text = re.sub(r'([\u0e00-\u0e7f]{2,6})\1{2,}', r'\1', text)
    return " ".join(text.split()).strip()


def translate(mt: MTModel, text: str, src_lang: str) -> str:
    """
    Translate text with a CTranslate2 HPLT Marian model.

    For Thai source text, word-tokenise first so the SPM encoder gets
    space-separated input rather than a continuous character stream.

    HPLT models converted with ct2-opus-mt-converter handle </s> internally —
    do NOT append it manually.
    """
    if not text.strip():
        return ""

    if src_lang == "th":
        text = _tokenize_thai_for_mt(text)

    results = []
    for sent in (_split_sentences(text) or [text]):
        toks = mt.sp.encode(sent, out_type=str)
        res = mt.translator.translate_batch(
            [toks],
            beam_size=CT2_BEAM_SIZE,
            max_decoding_length=CT2_MAX_DECODING_LEN,
            no_repeat_ngram_size=CT2_NO_REPEAT_NGRAM,
            repetition_penalty=CT2_REPETITION_PENALTY,
        )
        out = [t for t in res[0].hypotheses[0]
               if t not in {"</s>", "<s>", "<pad>", "<unk>"}]
        results.append(mt.sp.decode(out).strip())
    return _clean(" ".join(results))


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
def _aplay(wav_path: str):
    """Play a WAV file through the wm8960 soundcard via aplay."""
    env = os.environ.copy()
    env.pop("ALSA_CONFIG_PATH", None)
    result = subprocess.run(
        ["aplay", "-D", APLAY_DEVICE, wav_path],
        capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        print(f"  [TTS] aplay stderr: {result.stderr.strip()}")
    else:
        print("  [TTS] Playback complete.")


def _speak_thai(thai_tts, text: str):
    """
    Synthesise Thai text with PyThaiTTS and play via aplay.
    PyThaiTTS writes directly to a WAV file; we then play that file.
    """
    if thai_tts is None:
        print("  [TTS] PyThaiTTS not available — skipping Thai speech.")
        return
    try:
        kwargs = {"filename": TTS_OUT_FILE}
        # vachana model supports speaker_idx; others silently ignore it
        if THAI_TTS_MODEL == "vachana":
            kwargs["speaker_idx"] = THAI_TTS_SPEAKER

        out_path = thai_tts.tts(text, **kwargs)
        print(f"  [TTS-TH] Written: {out_path}")
        _aplay(TTS_OUT_FILE)
    except Exception as e:
        import traceback
        print(f"  [TTS-TH] Error: {e}")
        traceback.print_exc()


def _speak_english(text: str):
    """
    Synthesise English text with espeak-ng, capture WAV output and play via
    aplay so the audio goes through the wm8960 soundcard (not the default
    ALSA device that espeak-ng would use directly).
    """
    try:
        env = os.environ.copy()
        env.pop("ALSA_CONFIG_PATH", None)

        # espeak-ng -w writes a 16-bit mono WAV at its native rate (usually 22050 Hz)
        result = subprocess.run(
            [
                "espeak-ng",
                "-v", ESPEAK_EN_VOICE,
                "-s", str(ESPEAK_RATE),
                "-w", TTS_OUT_FILE,
                text,
            ],
            capture_output=True, text=True, env=env
        )
        if result.returncode != 0:
            print(f"  [TTS-EN] espeak-ng stderr: {result.stderr.strip()}")
            return
        print(f"  [TTS-EN] espeak-ng wrote {os.path.getsize(TTS_OUT_FILE)} bytes")
        _aplay(TTS_OUT_FILE)
    except FileNotFoundError:
        print("  [TTS-EN] espeak-ng not found — install with: sudo apt install espeak-ng")
    except Exception as e:
        import traceback
        print(f"  [TTS-EN] Error: {e}")
        traceback.print_exc()


def speak(app: AppState, text: str, lang: str):
    """Dispatch TTS to the correct engine based on target language."""
    if not ENABLE_TTS or not text:
        return
    print(f"  [TTS] lang={lang}, text={text!r}")
    if lang == "th":
        _speak_thai(app.thai_tts, text)
    else:
        _speak_english(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    board = WhisPlayBoard()
    board.set_backlight(50)

    images = {
        "idle":       _load_image(board, os.path.join(IMGS_DIR, "passive.jpg")),
        "listening":  _load_image(board, os.path.join(IMGS_DIR, "listening.jpg")),
        "processing": _load_image(board, os.path.join(IMGS_DIR, "talking.jpg")),
        "loading":    _load_image(board, os.path.join(IMGS_DIR, "loading.jpg")),
    }

    if images["loading"]:
        board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, images["loading"])
        print("[Display] Loading image displayed")

    in_device = _find_sd_input_device()

    # Set speaker volume
    card_name = os.environ.get("WM8960_CARD_NAME", "wm8960soundcard")
    try:
        env = os.environ.copy()
        env.pop("ALSA_CONFIG_PATH", None)
        subprocess.run(
            ["amixer", "-D", f"hw:{card_name}", "sset", "Speaker", "127"],
            check=True, capture_output=True, text=True, env=env
        )
        print("[Audio] Speaker volume set to 127")
    except Exception as e:
        print(f"[Audio] Could not set speaker volume: {e}")

    print(f"[Audio] output device: {APLAY_DEVICE}")

    app            = init_app()
    state          = [State.IDLE]
    eng_to_th      = [True]   # True = EN speaker goes first (EN->TH)
    last_activity  = [0.0]
    press_time     = [0.0]
    stop_recording = threading.Event()
    aborted        = [False]

    def set_state(new_state: State):
        state[0] = new_state
        label = {
            State.IDLE:       "IDLE       — waiting for button",
            State.LISTENING:  "LISTENING  — recording ...",
            State.PROCESSING: "PROCESSING — transcribing, translating & speaking ...",
        }[new_state]
        print(f"\n[STATE] {label}")
        update_display(board, new_state, images)

    def on_press():
        if state[0] == State.IDLE:
            press_time[0] = time.time()
        elif state[0] == State.LISTENING:
            print("[BTN] Press -> stopping recording")
            stop_recording.set()

    def on_release():
        if state[0] == State.IDLE:
            duration = time.time() - press_time[0]
            if duration >= SHUTOFF_DURATION:
                print(f"[BTN] Very long press ({duration:.2f}s) -> Shutting down")
                board.cleanup()
                subprocess.run(["sudo", "shutdown", "-h", "now"])
                return
            elif duration >= HOLD_DURATION:
                eng_to_th[0] = True
                print(f"[BTN] Long press ({duration:.2f}s) -> English speaker first (EN->TH)")
            else:
                eng_to_th[0] = False
                print(f"[BTN] Short press ({duration:.2f}s) -> Thai speaker first (TH->EN)")

            last_activity[0] = time.time()
            stop_recording.clear()
            set_state(State.LISTENING)
            threading.Thread(target=_recording_thread, daemon=True).start()

    def _recording_thread():
        record_until_button(stop_recording, in_device=in_device)

        if aborted[0]:
            aborted[0] = False
            return

        set_state(State.PROCESSING)

        is_en_first = eng_to_th[0]
        direction   = "EN->TH" if is_en_first else "TH->EN"
        vosk_model  = app.vosk_en if is_en_first else app.vosk_th
        mt          = app.en_th   if is_en_first else app.th_en
        src_lang    = "en"        if is_en_first else "th"
        tgt_lang    = "th"        if is_en_first else "en"

        raw  = _transcribe_wav(vosk_model, REC_FILE)
        text = _dedup_stt(raw)
        print(f"  Recognised ({src_lang}): {text}"
              + (f"  (raw: {raw})" if raw != text else ""))

        if not text:
            print("  [STT] Nothing recognised — restarting listener")
            last_activity[0] = time.time()
            stop_recording.clear()
            set_state(State.LISTENING)
            threading.Thread(target=_recording_thread, daemon=True).start()
            return

        t0  = time.time()
        out = translate(mt, text, src_lang)
        print(f"  [{direction}] {text}\n  ->  {out}  [{time.time()-t0:.2f}s]")

        speak(app, out, tgt_lang)

        # Swap direction for the next turn
        eng_to_th[0]     = not eng_to_th[0]
        last_activity[0] = time.time()

        stop_recording.clear()
        set_state(State.LISTENING)
        threading.Thread(target=_recording_thread, daemon=True).start()

    board.on_button_press(on_press)
    board.on_button_release(on_release)

    set_state(State.IDLE)
    print("Button: VERY LONG press (>=2.0s) = shutdown")
    print("        LONG  press (>=0.4s) = English speaker first (EN->TH)")
    print("        SHORT press (<0.4s)  = Thai speaker first (TH->EN)")

    try:
        while True:
            if state[0] != State.IDLE:
                if time.time() - last_activity[0] > CONVO_TIMEOUT:
                    print("\n[INACTIVITY] Returning to IDLE")
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