"""
WhisPlay HAT Translator — EN <-> ZH
Vosk STT  +  CTranslate2 OPUS MT  +  Piper TTS

Hardware: WhisPlay HAT (Raspberry Pi Zero 2W)
  - SHORT press while IDLE  = start ZH->EN
  - LONG press  while IDLE  = start EN->ZH  (hold >= 1s)
  - Any press while LISTENING = stop recording -> translate -> speak -> next turn
  - 15s inactivity timeout = return to IDLE

LAZY LOADING: Only one Vosk + one MT model in RAM at a time (fits 512MB).

Install deps:
    pip install vosk ctranslate2 sentencepiece sounddevice numpy piper-tts
    sudo apt install espeak-ng
"""

from __future__ import annotations

import gc
import io
import json
import os
import re
import sys
import time
import threading
import wave
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

try:
    sys.path.append("/home/teamnfg/EDD-Codebase/Driver")
    from WhisPlay import WhisPlayBoard
    WHISPLAY_AVAILABLE = True
except Exception as e:
    WHISPLAY_AVAILABLE = False
    print(f"[WARN] WhisPlay driver not available: {e} — running in headless mode")


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
CT2_COMPUTE_TYPE       = "int8"
CT2_BEAM_SIZE          = 2
CT2_MAX_DECODING_LEN   = 256
CT2_NO_REPEAT_NGRAM    = 3
CT2_REPETITION_PENALTY = 1.15
CT2_INTER_THREADS      = 1
CT2_INTRA_THREADS      = 2

MIC_SAMPLE_RATE  = 16_000
MIC_CHANNELS     = 1
MIC_BLOCK_SEC    = 0.25

HOLD_DURATION    = 1.0    # seconds — long press threshold
CONVO_TIMEOUT    = 15.0   # seconds — inactivity -> back to IDLE
MAX_RECORD_SEC   = 30

ENABLE_TTS = True


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model cache — only one model of each type in RAM at a time
# ─────────────────────────────────────────────────────────────────────────────
class LazyModelCache:
    def __init__(self):
        self._vosk_key  = None
        self._vosk      = None
        self._mt_key    = None
        self._mt        = None
        self._piper_key = None
        self._piper     = None

    def get_vosk(self, model_dir: str) -> vosk.Model:
        if self._vosk_key == model_dir:
            return self._vosk
        print(f"  [Cache] Loading Vosk: {os.path.basename(model_dir)}")
        self._vosk = None
        gc.collect()
        self._vosk = vosk.Model(model_dir)
        self._vosk_key = model_dir
        return self._vosk

    def get_mt(self, model_dir: str):
        if self._mt_key == model_dir:
            return self._mt
        print(f"  [Cache] Loading MT: {os.path.basename(model_dir)}")
        self._mt = None
        gc.collect()
        self._mt = _load_ct2(model_dir)
        self._mt_key = model_dir
        return self._mt

    def get_piper(self, onnx_path: str):
        if not PIPER_AVAILABLE or not os.path.isfile(onnx_path):
            return None
        if self._piper_key == onnx_path:
            return self._piper
        print(f"  [Cache] Loading Piper: {os.path.basename(onnx_path)}")
        self._piper = None
        gc.collect()
        json_path = onnx_path + ".json"
        try:
            self._piper = PiperVoice.load(
                onnx_path,
                config_path=json_path if os.path.isfile(json_path) else None,
                use_cuda=False,
            )
            self._piper_key = onnx_path
        except Exception as e:
            print(f"  [Piper] load error: {e}")
            self._piper = None
            self._piper_key = None
        return self._piper

    def unload_all(self):
        self._vosk = self._mt = self._piper = None
        self._vosk_key = self._mt_key = self._piper_key = None
        gc.collect()
        print("  [Cache] All models unloaded.")


_cache = LazyModelCache()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def _load_ct2(model_dir: str):
    for f in ("model.bin", "source.spm", "target.spm"):
        p = os.path.join(model_dir, f)
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    translator = ctranslate2.Translator(
        model_dir,
        compute_type=CT2_COMPUTE_TYPE,
        inter_threads=CT2_INTER_THREADS,
        intra_threads=CT2_INTRA_THREADS,
        max_queued_batches=1,
    )
    sp_src = spm.SentencePieceProcessor()
    sp_src.load(os.path.join(model_dir, "source.spm"))
    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.load(os.path.join(model_dir, "target.spm"))
    return translator, sp_src, sp_tgt


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_image(board, filepath: str):
    try:
        from PIL import Image
        img = Image.open(filepath).convert("RGB")
        img = img.resize((board.LCD_WIDTH, board.LCD_HEIGHT))
        data = []
        for y in range(board.LCD_HEIGHT):
            for x in range(board.LCD_WIDTH):
                r, g, b = img.getpixel((x, y))
                rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
                data.extend([(rgb565 >> 8) & 0xFF, rgb565 & 0xFF])
        return data
    except Exception as e:
        print(f"  [Display] Could not load image {filepath}: {e}")
        return None


def update_display(board, state: State, images: dict):
    if board is None:
        return
    img = images.get(state.name.lower())
    if img:
        try:
            board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img)
        except Exception as e:
            print(f"  [Display] Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
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
def record_until_button(stop_event: threading.Event) -> str:
    block_size = int(MIC_SAMPLE_RATE * MIC_BLOCK_SEC)
    max_blocks = int(MAX_RECORD_SEC / MIC_BLOCK_SEC)
    frames = []

    print("  [MIC] Recording — press button to stop ...")
    with sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=MIC_CHANNELS,
                        dtype="int16", blocksize=block_size) as stream:
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


def translate(translator, sp_src, sp_tgt, text: str) -> str:
    if not text.strip():
        return ""
    results = []
    for sent in (_split_sentences(text) or [text]):
        toks = sp_src.encode(sent, out_type=str)
        if sp_src.piece_to_id("</s>") != sp_src.unk_id():
            toks = toks + ["</s>"]
        res = translator.translate_batch(
            [toks],
            beam_size=CT2_BEAM_SIZE,
            max_decoding_length=CT2_MAX_DECODING_LEN,
            no_repeat_ngram_size=CT2_NO_REPEAT_NGRAM,
            repetition_penalty=CT2_REPETITION_PENALTY,
        )
        out = [t for t in res[0].hypotheses[0] if t not in {"</s>", "<s>", "<pad>"}]
        results.append(sp_tgt.decode(out).strip())
    return _clean(" ".join(results))


# ─────────────────────────────────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────────────────────────────────
def _piper_play(voice, text: str):
    if hasattr(voice, "synthesize_stream_raw"):
        rate   = voice.config.sample_rate
        stream = sd.OutputStream(samplerate=rate, channels=1, dtype="int16")
        stream.start()
        for audio_bytes in voice.synthesize_stream_raw(text):
            stream.write(np.frombuffer(audio_bytes, dtype=np.int16))
        stream.stop()
        stream.close()
        return
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
        sd.play(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0,
                samplerate=rate)
        sd.wait()
        return
    raise RuntimeError("No supported synthesize method on PiperVoice.")


def speak(text: str, lang: str):
    if not ENABLE_TTS or not text:
        return
    onnx_path = VOICE_ZH if lang == "zh" else VOICE_EN
    voice = _cache.get_piper(onnx_path)
    if voice:
        try:
            _piper_play(voice, text)
            return
        except Exception as e:
            print(f"  [TTS] Piper error: {e}")
    print(f"  [TTS] No TTS available for lang={lang}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Board init ────────────────────────────────────────────────────────────
    board = None
    images = {}
    if WHISPLAY_AVAILABLE:
        try:
            board = WhisPlayBoard()
            board.set_backlight(50)
            images["idle"]       = _load_image(board, os.path.join(IMGS_DIR, "passive.jpg"))
            images["listening"]  = _load_image(board, os.path.join(IMGS_DIR, "recording.jpg"))
            images["processing"] = _load_image(board, os.path.join(IMGS_DIR, "playing.jpg"))
            print("[INFO] WhisPlay board initialized.")
        except Exception as e:
            print(f"[WARN] WhisPlay board init failed: {e} — running in headless mode")
            board = None

    # ── Shared state ──────────────────────────────────────────────────────────
    state          = State.IDLE
    eng_to_cn      = True
    press_time     = 0.0
    last_activity  = 0.0
    stop_recording = threading.Event()

    def set_state(new_state: State):
        nonlocal state
        state = new_state
        labels = {
            State.IDLE:       "IDLE       — waiting for button",
            State.LISTENING:  "LISTENING  — recording ...",
            State.PROCESSING: "PROCESSING — translating & speaking ...",
        }
        print(f"\n[STATE] {labels[new_state]}")
        update_display(board, new_state, images)

    # ── Button callbacks ──────────────────────────────────────────────────────
    # WhisPlay fires on_button_press on HIGH (button down)
    # and on_button_release on LOW (button up) via GPIO.BOTH
    def on_press():
        nonlocal press_time
        press_time = time.time()

    def on_release():
        nonlocal eng_to_cn, last_activity

        duration = time.time() - press_time

        if state == State.IDLE:
            eng_to_cn = duration >= HOLD_DURATION
            direction = "EN->ZH" if eng_to_cn else "ZH->EN"
            print(f"[BTN] {'Long' if eng_to_cn else 'Short'} press -> {direction}")
            last_activity = time.time()
            stop_recording.clear()
            set_state(State.LISTENING)
            threading.Thread(target=_recording_thread, daemon=True).start()

        elif state == State.LISTENING:
            print("[BTN] Press -> stopping recording")
            stop_recording.set()

    # ── Recording + processing thread ────────────────────────────────────────
    def _recording_thread():
        nonlocal eng_to_cn, last_activity

        record_until_button(stop_recording)
        set_state(State.PROCESSING)

        vosk_dir  = VOSK_EN_DIR   if eng_to_cn else VOSK_ZH_DIR
        mt_dir    = CT2_EN_ZH_DIR if eng_to_cn else CT2_ZH_EN_DIR
        tgt_lang  = "zh"          if eng_to_cn else "en"
        direction = "EN->ZH"      if eng_to_cn else "ZH->EN"

        print(f"  [Load] Vosk {direction.split('->')[0]} ...")
        vosk_model = _cache.get_vosk(vosk_dir)

        raw  = _transcribe_wav(vosk_model, REC_FILE)
        text = _dedup_stt(raw)
        print(f"  Recognised: {text}" + (f"  (raw: {raw})" if raw != text else ""))

        if not text:
            print("  Nothing recognised — returning to LISTEN.")
            last_activity = time.time()
            stop_recording.clear()
            set_state(State.LISTENING)
            threading.Thread(target=_recording_thread, daemon=True).start()
            return

        print(f"  [Load] MT {direction} ...")
        translator, sp_src, sp_tgt = _cache.get_mt(mt_dir)
        t0  = time.time()
        out = translate(translator, sp_src, sp_tgt, text)
        print(f"  [{direction}] {text}\n  ->  {out}  [{time.time()-t0:.2f}s]")

        speak(out, tgt_lang)

        eng_to_cn = not eng_to_cn
        last_activity = time.time()
        stop_recording.clear()
        set_state(State.LISTENING)
        threading.Thread(target=_recording_thread, daemon=True).start()

    # ── Register button callbacks ─────────────────────────────────────────────
    if board:
        board.on_button_press(on_press)
        board.on_button_release(on_release)
    else:
        # Headless keyboard fallback — press ENTER to simulate button
        print("[INFO] No board — press ENTER to simulate button press/release.")
        def _keyboard_thread():
            while True:
                input()
                on_press()
                sleep(0.1)
                on_release()
        threading.Thread(target=_keyboard_thread, daemon=True).start()

    set_state(State.IDLE)
    print("Button: LONG press  (>=1s) = EN->ZH")
    print("        SHORT press (<1s)  = ZH->EN")
    print()

    # ── Main loop — timeout watchdog ─────────────────────────────────────────
    try:
        while True:
            if state != State.IDLE:
                if time.time() - last_activity > CONVO_TIMEOUT:
                    print("\n[TIMEOUT] 15s inactivity — returning to IDLE.")
                    stop_recording.set()
                    sleep(0.5)
                    _cache.unload_all()
                    set_state(State.IDLE)
            sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting ...")
        stop_recording.set()
        _cache.unload_all()
        if board:
            board.cleanup()


if __name__ == "__main__":
    main()