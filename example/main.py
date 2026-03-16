"""
WhisPlay HAT Translator — EN <-> ZH  (Raspberry Pi Zero 2W)
Vosk STT  +  CTranslate2 OPUS MT  +  Piper TTS

Hardware: WhisPlay HAT
  - SHORT press = start ZH→EN conversation
  - LONG  press (≥1s) = start EN→ZH conversation
  - Press again while recording = stop and translate
  - 15s inactivity timeout = back to IDLE

Run with:
    bash run_translator.sh

REQUIRED — use SMALL vosk models only (large models will OOM crash):
    models/vosk-en  →  vosk-model-small-en-us-0.15   (~40 MB)
    models/vosk-cn  →  vosk-model-small-cn-0.22       (~40 MB)
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import sys
import time
import threading
import wave
from dataclasses import dataclass
from enum import Enum
from typing import List

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

sys.path.append(os.path.abspath("../Driver"))
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
VOSK_EN_DIR   = os.path.join(MODELS_DIR, "vosk-en")
VOSK_ZH_DIR   = os.path.join(MODELS_DIR, "vosk-cn")
CT2_EN_ZH_DIR = os.path.join(MODELS_DIR, "opus-en-zh-ct2")
CT2_ZH_EN_DIR = os.path.join(MODELS_DIR, "opus-zh-en-ct2")
VOICE_EN      = os.path.join(VOICES_DIR, "en_US-lessac-low.onnx")
VOICE_ZH      = os.path.join(VOICES_DIR, "zh_CN-huayan-x_low.onnx")

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

# CTranslate2 — pin threads to avoid thrashing Pi Zero 2W's 4 slow cores.
# 1 inter + 2 intra leaves 1-2 cores free for OS/GPIO/audio.
CT2_INTER_THREADS      = 1
CT2_INTRA_THREADS      = 2
CT2_COMPUTE_TYPE       = "int8"
CT2_BEAM_SIZE          = 2
CT2_MAX_DECODING_LEN   = 256
CT2_NO_REPEAT_NGRAM    = 3
CT2_REPETITION_PENALTY = 1.15

MIC_SAMPLE_RATE  = 16_000
MIC_CHANNELS     = 1
MIC_BLOCK_SEC    = 0.25
MAX_RECORD_SEC   = 30

HOLD_DURATION  = 1.0   # seconds — long press threshold
CONVO_TIMEOUT  = 15.0  # seconds inactivity → IDLE

ENABLE_TTS = True

# Resolved at startup by _find_wm8960_device()
SD_INPUT_DEVICE  = None
SD_OUTPUT_DEVICE = None


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
# ALSA / sounddevice device discovery
# ─────────────────────────────────────────────────────────────────────────────
def _find_wm8960_device():
    """
    Return (input_idx, output_idx) for the wm8960 sound card.
    Falls back to (None, None) so sounddevice uses its own default.
    sounddevice/PortAudio ignores the AUDIODEV env var, so we must
    pass the device index explicitly on every open() call.
    """
    try:
        devices = sd.query_devices()
        in_idx = out_idx = None
        for i, d in enumerate(devices):
            name = d.get("name", "").lower()
            if "wm8960" in name or "seeed" in name:
                if d.get("max_input_channels", 0) > 0 and in_idx is None:
                    in_idx = i
                if d.get("max_output_channels", 0) > 0 and out_idx is None:
                    out_idx = i
        if in_idx is not None:
            print(f"[AUDIO] input  → device {in_idx}: {devices[in_idx]['name']}")
        if out_idx is not None:
            print(f"[AUDIO] output → device {out_idx}: {devices[out_idx]['name']}")
        if in_idx is None and out_idx is None:
            print("[AUDIO] wm8960 not found in sounddevice list — using system default")
        return in_idx, out_idx
    except Exception as e:
        print(f"[AUDIO] Device discovery error: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def _load_ct2(model_dir: str) -> MTModel:
    for f in ("model.bin", "source.spm", "target.spm"):
        p = os.path.join(model_dir, f)
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Missing model file: {p}\n"
                "Ensure you are using CTranslate2-converted OPUS-MT models."
            )
    # inter_threads=1, intra_threads=2 prevents thread over-subscription on
    # the Pi Zero 2W's 4 cores (leaves headroom for OS + GPIO + audio).
    translator = ctranslate2.Translator(
        model_dir,
        compute_type=CT2_COMPUTE_TYPE,
        inter_threads=CT2_INTER_THREADS,
        intra_threads=CT2_INTRA_THREADS,
    )
    sp_src = spm.SentencePieceProcessor()
    sp_src.load(os.path.join(model_dir, "source.spm"))
    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.load(os.path.join(model_dir, "target.spm"))
    return MTModel(translator, sp_src, sp_tgt)


def _load_piper_voice(onnx_path: str):
    if not PIPER_AVAILABLE:
        return None
    if not os.path.isfile(onnx_path):
        print(f"  [Piper] voice file not found: {onnx_path}")
        return None
    json_path = onnx_path + ".json"
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
    """
    Load all models sequentially. On Pi Zero 2W each step takes 5-30s.
    Total startup ~60-120s is normal — do not kill the process.
    """
    print("Loading Vosk EN  (must be vosk-model-small-en-us, ~40MB) ...")
    vosk_en = vosk.Model(VOSK_EN_DIR)

    print("Loading Vosk ZH  (must be vosk-model-small-cn, ~40MB) ...")
    vosk_zh = vosk.Model(VOSK_ZH_DIR)

    print("Loading EN->ZH translation model ...")
    en_zh = _load_ct2(CT2_EN_ZH_DIR)

    print("Loading ZH->EN translation model ...")
    zh_en = _load_ct2(CT2_ZH_EN_DIR)

    piper_en = piper_zh = None
    if PIPER_AVAILABLE:
        print("Loading Piper EN voice ...")
        piper_en = _load_piper_voice(VOICE_EN)
        print("Loading Piper ZH voice ...")
        piper_zh = _load_piper_voice(VOICE_ZH)
        loaded = [n for n, v in [("EN", piper_en), ("ZH", piper_zh)] if v]
        print(f"Piper voices loaded: {loaded if loaded else 'none — TTS will be silent'}")

    print("\nAll models loaded.\n")
    return AppState(vosk_en, vosk_zh, en_zh, zh_en, piper_en, piper_zh)


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ALL LCD/SPI writes must go through _display_queue to the single
# _display_worker thread. SPI is not thread-safe; the GPIO interrupt thread
# and main loop would otherwise collide on the bus and cause GPIO errors.
# ─────────────────────────────────────────────────────────────────────────────
_display_queue: queue.Queue = queue.Queue()


def _load_image_rgb565(board, filepath: str):
    """
    Load image and convert to RGB565 byte list using numpy vectorisation.
    This is ~50x faster than per-pixel getpixel() calls — important on
    Pi ARM at startup when loading 3 images.
    """
    try:
        from PIL import Image
        img = Image.open(filepath).convert("RGB")
        img = img.resize((board.LCD_WIDTH, board.LCD_HEIGHT), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint16)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        high = (rgb565 >> 8).astype(np.uint8)
        low  = (rgb565 & 0xFF).astype(np.uint8)
        out = np.empty((board.LCD_HEIGHT, board.LCD_WIDTH, 2), dtype=np.uint8)
        out[:, :, 0] = high
        out[:, :, 1] = low
        return out.flatten().tolist()
    except Exception as e:
        print(f"  [Display] Could not load '{filepath}': {e}")
        return None


def _display_worker(board, images: dict):
    """The ONLY thread that calls board SPI methods."""
    while True:
        item = _display_queue.get()
        if item is None:
            _display_queue.task_done()
            break
        cmd, payload = item
        try:
            if cmd == "image":
                img = images.get(payload)
                if img:
                    board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img)
                else:
                    print(f"  [Display] No image for key '{payload}'")
            elif cmd == "rgb":
                board.set_rgb(*payload)
            elif cmd == "backlight":
                board.set_backlight(payload)
        except Exception as e:
            print(f"  [Display] Error on '{cmd}': {e}")
        finally:
            _display_queue.task_done()


def display_image(key: str):
    _display_queue.put(("image", key))

def display_rgb(r: int, g: int, b: int):
    _display_queue.put(("rgb", (r, g, b)))

def display_backlight(v: int):
    _display_queue.put(("backlight", v))


# ─────────────────────────────────────────────────────────────────────────────
# Recording
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.abs(audio.astype(np.float32)).max()
    if peak < 1:
        return audio
    scale = (32767 * 0.9) / peak
    if scale <= 1.0:
        return audio
    return np.clip(audio.astype(np.float32) * scale, -32768, 32767).astype(np.int16)


def record_until_stopped(stop_event: threading.Event):
    """
    Record mic audio into REC_FILE until stop_event is set or MAX_RECORD_SEC.
    latency='high' is more stable than 'low' on Pi Zero 2W with PortAudio.
    """
    block_size = int(MIC_SAMPLE_RATE * MIC_BLOCK_SEC)
    max_blocks = int(MAX_RECORD_SEC / MIC_BLOCK_SEC)
    frames = []

    print("  [MIC] Recording — press button to stop ...")
    try:
        with sd.InputStream(
            device=SD_INPUT_DEVICE,
            samplerate=MIC_SAMPLE_RATE,
            channels=MIC_CHANNELS,
            dtype="int16",
            blocksize=block_size,
            latency="high",
        ) as stream:
            for _ in range(max_blocks):
                if stop_event.is_set():
                    break
                block, _ = stream.read(block_size)
                frames.append(block.copy())
    except Exception as e:
        print(f"  [MIC] InputStream error: {e}")

    print("  [MIC] Stopped.")
    if not frames:
        frames = [np.zeros((block_size, MIC_CHANNELS), dtype=np.int16)]

    audio = np.concatenate(frames, axis=0)
    with wave.open(REC_FILE, "wb") as wf:
        wf.setnchannels(MIC_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(MIC_SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


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
    parts = []
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
    n = len(words)
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
# ─────────────────────────────────────────────────────────────────────────────
def _piper_play(voice, text: str):
    """
    Play Piper TTS through the wm8960 speaker.
    Tries streaming first (lower RAM peak), falls back to file-based.
    SD_OUTPUT_DEVICE passed explicitly — PortAudio ignores AUDIODEV env var.
    """
    if hasattr(voice, "synthesize_stream_raw"):
        rate = voice.config.sample_rate
        try:
            stream = sd.OutputStream(
                device=SD_OUTPUT_DEVICE,
                samplerate=rate,
                channels=1,
                dtype="int16",
                latency="high",
            )
            stream.start()
            for audio_bytes in voice.synthesize_stream_raw(text):
                stream.write(np.frombuffer(audio_bytes, dtype=np.int16))
            stream.stop()
            stream.close()
            return
        except Exception as e:
            print(f"  [TTS] stream error ({e}) — trying file method")

    if hasattr(voice, "synthesize"):
        out_wav = os.path.join(DATA_DIR, "tts_out.wav")
        rate    = getattr(getattr(voice, "config", None), "sample_rate", 22050)
        with wave.open(out_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            voice.synthesize(text, wf)
        with wave.open(out_wav, "rb") as wf:
            data = wf.readframes(wf.getnframes())
        sd.play(
            np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0,
            samplerate=rate,
            device=SD_OUTPUT_DEVICE,
        )
        sd.wait()
        return

    raise RuntimeError("PiperVoice has no synthesize_stream_raw or synthesize method.")


def speak(app: AppState, text: str, lang: str):
    if not ENABLE_TTS or not text:
        return
    voice = app.piper_zh if lang == "zh" else app.piper_en
    if voice:
        try:
            _piper_play(voice, text)
        except Exception as e:
            print(f"  [TTS] Error: {e}")
    else:
        print(f"  [TTS] No voice loaded for lang='{lang}'")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global SD_INPUT_DEVICE, SD_OUTPUT_DEVICE

    # 1. Find wm8960 device indices for sounddevice
    SD_INPUT_DEVICE, SD_OUTPUT_DEVICE = _find_wm8960_device()

    # 2. Load models (slow on Pi Zero 2W — 60-120s total is normal)
    app = init_app()

    # 3. Init WhisPlay hardware
    board = WhisPlayBoard()
    board.set_backlight(50)

    # 4. Load display images (numpy-vectorised, fast)
    print("Loading display images ...")
    images = {
        "idle":       _load_image_rgb565(board, os.path.join(IMGS_DIR, "passive.jpg")),
        "listening":  _load_image_rgb565(board, os.path.join(IMGS_DIR, "listening.jpg")),
        "processing": _load_image_rgb565(board, os.path.join(IMGS_DIR, "talking.jpg")),
    }

    # 5. Start single display thread — the ONLY owner of SPI
    disp_thread = threading.Thread(
        target=_display_worker, args=(board, images),
        daemon=True, name="display"
    )
    disp_thread.start()

    # 6. Shared state — only written by the main loop below
    state         = State.IDLE
    eng_to_cn     = True
    last_activity = time.time()
    stop_rec      = threading.Event()
    rec_thread    = [None]   # list so nested functions can rebind

    # Button callbacks: ONLY enqueue events — never touch state or SPI
    press_time = [0.0]
    event_queue: queue.Queue = queue.Queue()

    def on_press():
        press_time[0] = time.time()
        event_queue.put(("press", press_time[0]))

    def on_release():
        event_queue.put(("release", time.time()))

    board.on_button_press(on_press)
    board.on_button_release(on_release)

    def start_recording():
        stop_rec.clear()
        t = threading.Thread(
            target=record_until_stopped, args=(stop_rec,),
            daemon=True, name="recorder"
        )
        t.start()
        rec_thread[0] = t

    def stop_recording():
        stop_rec.set()
        if rec_thread[0] is not None:
            rec_thread[0].join(timeout=3.0)
            rec_thread[0] = None

    def process_audio() -> bool:
        """STT → translate → TTS speak. Returns True if speech recognised."""
        nonlocal eng_to_cn, last_activity

        vosk_model = app.vosk_en if eng_to_cn else app.vosk_zh
        mt         = app.en_zh   if eng_to_cn else app.zh_en
        tgt_lang   = "zh"        if eng_to_cn else "en"
        direction  = "EN->ZH"    if eng_to_cn else "ZH->EN"

        print(f"  [STT] Transcribing ({direction}) ...")
        raw  = _transcribe_wav(vosk_model, REC_FILE)
        text = _dedup_stt(raw)
        if raw != text:
            print(f"  Recognised (deduped): {text}  (raw: {raw})")
        else:
            print(f"  Recognised: {text}")

        if not text:
            print("  Nothing recognised — try again.")
            return False

        t0  = time.time()
        out = translate(mt, text)
        print(f"  [{direction}] {text}\n  ->  {out}  [{time.time()-t0:.1f}s]")

        speak(app, out, tgt_lang)
        last_activity = time.time()
        eng_to_cn = not eng_to_cn  # swap direction for next speaker
        return True

    # 7. Initial display
    display_image("idle")
    display_rgb(0, 0, 0)
    print("Ready.")
    print("  LONG press  (≥1s) → EN→ZH")
    print("  SHORT press       → ZH→EN")
    print("  Press while recording → stop & translate")
    print(f"  {CONVO_TIMEOUT:.0f}s silence → back to IDLE\n")

    # 8. Main event loop
    try:
        while True:

            # Inactivity timeout watchdog
            if state != State.IDLE:
                if time.time() - last_activity > CONVO_TIMEOUT:
                    print("\n[TIMEOUT] Returning to IDLE.")
                    stop_recording()
                    state = State.IDLE
                    display_image("idle")
                    display_rgb(0, 0, 0)

            # Non-blocking event drain
            try:
                event, ts = event_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.05)
                continue

            # IDLE: wait for first button release to determine short/long press
            if state == State.IDLE:
                if event == "release":
                    duration = ts - press_time[0]
                    if duration >= HOLD_DURATION:
                        eng_to_cn = True
                        print("[BTN] Long press  → EN→ZH")
                    else:
                        eng_to_cn = False
                        print("[BTN] Short press → ZH→EN")
                    last_activity = time.time()
                    state = State.LISTENING
                    display_image("listening")
                    display_rgb(0, 0, 255)
                    start_recording()

            # LISTENING: next press stops recording and triggers processing
            elif state == State.LISTENING:
                if event == "press":
                    stop_recording()
                    last_activity = time.time()
                    state = State.PROCESSING
                    display_image("processing")
                    display_rgb(255, 128, 0)

                    # Blocks here during STT + translation + TTS.
                    # Intentional — simpler and safer than threading it on Pi.
                    process_audio()

                    # Return to LISTENING for the other speaker
                    last_activity = time.time()
                    state = State.LISTENING
                    display_image("listening")
                    display_rgb(0, 0, 255)
                    start_recording()

            # PROCESSING: button ignored (we're blocking in process_audio above)

    except KeyboardInterrupt:
        print("\nCtrl+C — shutting down ...")

    finally:
        stop_rec.set()
        _display_queue.put(None)   # poison pill → stop display thread
        _display_queue.join()
        try:
            board.set_backlight(0)
            board.cleanup()
        except Exception:
            pass
        print("Shutdown complete.")


if __name__ == "__main__":
    main()