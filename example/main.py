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
import subprocess
from enum import Enum
from time import sleep
from typing import List

import numpy as np
import sounddevice as sd
import vosk
import ctranslate2
import sentencepiece as spm
from PIL import Image

try:
    from piper.voice import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("[WARN] piper-tts not installed. Run: pip install piper-tts")

sys.path.append("/home/teamnfg/EDD-Codebase/Driver")
from WhisPlay import WhisPlayBoard


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
class State(Enum):
    IDLE = 0
    LISTENING = 1
    PROCESSING = 2


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DATA_DIR = os.path.join(HERE, "data")
VOICES_DIR = os.path.join(HERE, "voices")
IMGS_DIR = os.path.join(HERE, "imgs")

os.makedirs(DATA_DIR, exist_ok=True)

REC_FILE = os.path.join(DATA_DIR, "recorded_voice.wav")
TTS_OUT_FILE = os.path.join(DATA_DIR, "tts_out.wav")

VOSK_EN_DIR = os.path.join(MODELS_DIR, "vosk-en")
VOSK_ZH_DIR = os.path.join(MODELS_DIR, "vosk-cn")

CT2_EN_ZH_DIR = os.path.join(MODELS_DIR, "opus-en-zh-ct2")
CT2_ZH_EN_DIR = os.path.join(MODELS_DIR, "opus-zh-en-ct2")

VOICE_EN = os.path.join(VOICES_DIR, "en_US-lessac-low.onnx")
VOICE_ZH = os.path.join(VOICES_DIR, "zh_CN-huayan-x_low.onnx")

IDLE_IMG = os.path.join(IMGS_DIR, "passive.jpg")
LISTENING_IMG = os.path.join(IMGS_DIR, "recording.jpg")
PROCESSING_IMG = os.path.join(IMGS_DIR, "playing.jpg")


# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
CT2_COMPUTE_TYPE = "int8"
CT2_BEAM_SIZE = 2
CT2_MAX_DECODING_LEN = 256
CT2_NO_REPEAT_NGRAM = 3
CT2_REPETITION_PENALTY = 1.15
CT2_INTER_THREADS = 1
CT2_INTRA_THREADS = 2

MIC_SAMPLE_RATE = 16000
MIC_CHANNELS = 1
MIC_BLOCK_SEC = 0.25
MAX_RECORD_SEC = 30

HOLD_DURATION = 1.0
CONVO_TIMEOUT = 15.0

ENABLE_TTS = True


# ─────────────────────────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────────────────────────
board = None
images = {}

state = State.IDLE
eng_to_cn = True
press_time = 0.0
last_activity = 0.0
busy = False

stop_recording = threading.Event()
worker_thread = None
state_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Audio volume helper
# ─────────────────────────────────────────────────────────────────────────────
def set_wm8960_volume_stable(volume_level: str = "121"):
    card_name = "wm8960soundcard"
    control_name = "Speaker"
    device_arg = f"hw:{card_name}"

    command = [
        "amixer",
        "-D", device_arg,
        "sset",
        control_name,
        volume_level
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"[INFO] Set '{control_name}' volume to {volume_level} on card '{card_name}'.")
    except Exception as e:
        print(f"[WARN] Could not set wm8960 volume: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model cache
# ─────────────────────────────────────────────────────────────────────────────
class LazyModelCache:
    def __init__(self):
        self._vosk_key = None
        self._vosk = None

        self._mt_key = None
        self._mt = None

        self._piper_key = None
        self._piper = None

    def get_vosk(self, model_dir: str) -> vosk.Model:
        if self._vosk_key == model_dir and self._vosk is not None:
            return self._vosk

        print(f"  [Cache] Loading Vosk: {os.path.basename(model_dir)}")
        self._vosk = None
        gc.collect()

        self._vosk = vosk.Model(model_dir)
        self._vosk_key = model_dir
        return self._vosk

    def get_mt(self, model_dir: str):
        if self._mt_key == model_dir and self._mt is not None:
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

        if self._piper_key == onnx_path and self._piper is not None:
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
        self._vosk = None
        self._mt = None
        self._piper = None
        self._vosk_key = None
        self._mt_key = None
        self._piper_key = None
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
            raise FileNotFoundError(f"Missing required MT file: {p}")

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
def load_jpg_as_rgb565(filepath, screen_width, screen_height):
    img = Image.open(filepath).convert("RGB")
    original_width, original_height = img.size

    aspect_ratio = original_width / original_height
    screen_aspect_ratio = screen_width / screen_height

    if aspect_ratio > screen_aspect_ratio:
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)
        resized_img = img.resize((new_width, new_height))
        offset_x = (new_width - screen_width) // 2
        cropped_img = resized_img.crop((offset_x, 0, offset_x + screen_width, screen_height))
    else:
        new_width = screen_width
        new_height = int(new_width / aspect_ratio)
        resized_img = img.resize((new_width, new_height))
        offset_y = (new_height - screen_height) // 2
        cropped_img = resized_img.crop((0, offset_y, screen_width, offset_y + screen_height))

    pixel_data = []
    for y in range(screen_height):
        for x in range(screen_width):
            r, g, b = cropped_img.getpixel((x, y))
            rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
            pixel_data.extend([(rgb565 >> 8) & 0xFF, rgb565 & 0xFF])

    return pixel_data


def preload_images():
    global images
    images["idle"] = load_jpg_as_rgb565(IDLE_IMG, board.LCD_WIDTH, board.LCD_HEIGHT)
    images["listening"] = load_jpg_as_rgb565(LISTENING_IMG, board.LCD_WIDTH, board.LCD_HEIGHT)
    images["processing"] = load_jpg_as_rgb565(PROCESSING_IMG, board.LCD_WIDTH, board.LCD_HEIGHT)


def update_display(new_state: State):
    key = new_state.name.lower()
    img = images.get(key)
    if img is not None:
        board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img)

    if new_state == State.IDLE:
        board.set_rgb(0, 0, 255)
    elif new_state == State.LISTENING:
        board.set_rgb(255, 0, 0)
    elif new_state == State.PROCESSING:
        board.set_rgb(0, 255, 0)


def set_state(new_state: State):
    global state
    with state_lock:
        state = new_state
        print(f"\n[STATE] {new_state.name}")
        update_display(new_state)


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio

    peak = np.abs(audio.astype(np.float32)).max()
    if peak < 1:
        return audio

    scale = (32767 * 0.9) / peak
    if scale >= 1.0:
        return audio

    return np.clip(audio.astype(np.float32) * scale, -32768, 32767).astype(np.int16)


# ─────────────────────────────────────────────────────────────────────────────
# Recording
# ─────────────────────────────────────────────────────────────────────────────
def record_until_button(stop_event: threading.Event) -> str:
    block_size = int(MIC_SAMPLE_RATE * MIC_BLOCK_SEC)
    max_blocks = int(MAX_RECORD_SEC / MIC_BLOCK_SEC)
    frames = []

    print("  [MIC] Recording — press button again to stop ...")

    with sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=MIC_CHANNELS,
        dtype="int16",
        blocksize=block_size
    ) as stream:
        for _ in range(max_blocks):
            if stop_event.is_set():
                break
            block, _ = stream.read(block_size)
            frames.append(block.copy())

    print("  [MIC] Recording stopped.")

    if frames:
        audio = np.concatenate(frames, axis=0)
    else:
        audio = np.zeros((0, MIC_CHANNELS), dtype=np.int16)

    with wave.open(REC_FILE, "wb") as wf:
        wf.setnchannels(MIC_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(MIC_SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    return REC_FILE


# ─────────────────────────────────────────────────────────────────────────────
# STT
# ─────────────────────────────────────────────────────────────────────────────
_FILLER_RE = re.compile(r"\b(uh+|um+|hmm+|huh|mm+|ah+|er+)\b", re.IGNORECASE)


def _strip_fillers(text: str) -> str:
    return " ".join(_FILLER_RE.sub("", text).split())


def _transcribe_wav(model: vosk.Model, wav_path: str) -> str:
    with wave.open(wav_path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()

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
    parts = re.split(r"(?<=[。！？；.!?;])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def _clean(text: str) -> str:
    text = re.sub(r"([\u4e00-\u9fff\u3000-\u303f]{2,6})\1{2,}", r"\1", text)
    return " ".join(text.split()).strip()


def translate(translator, sp_src, sp_tgt, text: str) -> str:
    if not text.strip():
        return ""

    results = []
    for sent in (_split_sentences(text) or [text]):
        toks = sp_src.encode(sent, out_type=str)
        try:
            eos_id = sp_src.piece_to_id("</s>")
            if eos_id != sp_src.unk_id():
                toks = toks + ["</s>"]
        except Exception:
            pass

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
        rate = voice.config.sample_rate
        stream = sd.OutputStream(samplerate=rate, channels=1, dtype="int16")
        stream.start()

        try:
            for audio_bytes in voice.synthesize_stream_raw(text):
                stream.write(np.frombuffer(audio_bytes, dtype=np.int16))
        finally:
            stream.stop()
            stream.close()
        return

    if hasattr(voice, "synthesize"):
        rate = getattr(getattr(voice, "config", None), "sample_rate", 22050)

        with wave.open(TTS_OUT_FILE, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            voice.synthesize(text, wf)

        with wave.open(TTS_OUT_FILE, "rb") as wf:
            data = wf.readframes(wf.getnframes())

        sd.play(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0, samplerate=rate)
        sd.wait()
        return

    raise RuntimeError("Unsupported PiperVoice API")


def speak(text: str, lang: str):
    if not ENABLE_TTS or not text:
        return

    onnx_path = VOICE_ZH if lang == "zh" else VOICE_EN
    voice = _cache.get_piper(onnx_path)

    if voice is None:
        print(f"  [TTS] No Piper voice available for lang={lang}")
        return

    try:
        _piper_play(voice, text)
    except Exception as e:
        print(f"  [TTS] Piper error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Core worker
# ─────────────────────────────────────────────────────────────────────────────
def _conversation_worker(direction_en_to_zh: bool):
    global eng_to_cn, last_activity, busy, worker_thread

    try:
        stop_recording.clear()
        set_state(State.LISTENING)
        record_until_button(stop_recording)

        set_state(State.PROCESSING)

        vosk_dir = VOSK_EN_DIR if direction_en_to_zh else VOSK_ZH_DIR
        mt_dir = CT2_EN_ZH_DIR if direction_en_to_zh else CT2_ZH_EN_DIR
        tgt_lang = "zh" if direction_en_to_zh else "en"
        direction_label = "EN->ZH" if direction_en_to_zh else "ZH->EN"

        print(f"  [Load] Vosk {direction_label.split('->')[0]} ...")
        vosk_model = _cache.get_vosk(vosk_dir)

        raw = _transcribe_wav(vosk_model, REC_FILE)
        text = _dedup_stt(raw)
        print(f"  Recognized: {text}" + (f"  (raw: {raw})" if raw != text else ""))

        if not text:
            print("  Nothing recognized.")
            last_activity = time.time()
            set_state(State.IDLE)
            return

        print(f"  [Load] MT {direction_label} ...")
        translator, sp_src, sp_tgt = _cache.get_mt(mt_dir)

        t0 = time.time()
        out = translate(translator, sp_src, sp_tgt, text)
        dt = time.time() - t0

        print(f"  [{direction_label}] {text}\n  ->  {out}  [{dt:.2f}s]")

        if out:
            speak(out, tgt_lang)

        eng_to_cn = not direction_en_to_zh
        last_activity = time.time()
        set_state(State.LISTENING)

        # auto-listen next turn like your original translator
        stop_recording.clear()
        record_until_button(stop_recording)

        set_state(State.PROCESSING)

        vosk_dir = VOSK_EN_DIR if eng_to_cn else VOSK_ZH_DIR
        mt_dir = CT2_EN_ZH_DIR if eng_to_cn else CT2_ZH_EN_DIR
        tgt_lang = "zh" if eng_to_cn else "en"
        direction_label = "EN->ZH" if eng_to_cn else "ZH->EN"

        print(f"  [Load] Vosk {direction_label.split('->')[0]} ...")
        vosk_model = _cache.get_vosk(vosk_dir)

        raw = _transcribe_wav(vosk_model, REC_FILE)
        text = _dedup_stt(raw)
        print(f"  Recognized: {text}" + (f"  (raw: {raw})" if raw != text else ""))

        if not text:
            print("  Nothing recognized.")
            last_activity = time.time()
            set_state(State.IDLE)
            return

        print(f"  [Load] MT {direction_label} ...")
        translator, sp_src, sp_tgt = _cache.get_mt(mt_dir)

        t0 = time.time()
        out = translate(translator, sp_src, sp_tgt, text)
        dt = time.time() - t0

        print(f"  [{direction_label}] {text}\n  ->  {out}  [{dt:.2f}s]")

        if out:
            speak(out, tgt_lang)

        eng_to_cn = not eng_to_cn
        last_activity = time.time()
        set_state(State.IDLE)

    except Exception as e:
        print(f"[ERROR] Worker failed: {e}")
        set_state(State.IDLE)

    finally:
        busy = False
        worker_thread = None


# ─────────────────────────────────────────────────────────────────────────────
# Button callbacks
# ─────────────────────────────────────────────────────────────────────────────
def on_press():
    global press_time
    press_time = time.time()


def on_release():
    global busy, worker_thread, last_activity, eng_to_cn

    duration = time.time() - press_time

    with state_lock:
        current_state = state

    if current_state == State.IDLE:
        if busy:
            return

        direction_now = duration >= HOLD_DURATION
        eng_to_cn = direction_now
        print(f"[BTN] {'Long' if direction_now else 'Short'} press -> {'EN->ZH' if direction_now else 'ZH->EN'}")

        busy = True
        last_activity = time.time()

        worker_thread = threading.Thread(
            target=_conversation_worker,
            args=(direction_now,),
            daemon=True
        )
        worker_thread.start()

    elif current_state == State.LISTENING:
        print("[BTN] Stop recording")
        stop_recording.set()

    elif current_state == State.PROCESSING:
        print("[BTN] Ignored during processing")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global board, last_activity

    print("[INFO] Initializing WhisPlay board...")
    board = WhisPlayBoard()
    board.set_backlight(50)

    print("[INFO] Preloading images...")
    preload_images()

    print("[INFO] Setting audio volume...")
    set_wm8960_volume_stable("121")

    print("[INFO] Registering button callbacks...")
    board.on_button_press(on_press)
    board.on_button_release(on_release)

    last_activity = time.time()
    set_state(State.IDLE)

    print("Ready.")
    print("  LONG press  (>=1s): EN->ZH")
    print("  SHORT press (<1s):  ZH->EN")
    print("  Press again while recording to stop early")

    try:
        while True:
            with state_lock:
                current_state = state

            if current_state != State.IDLE and (time.time() - last_activity > CONVO_TIMEOUT):
                print(f"\n[TIMEOUT] {CONVO_TIMEOUT:.0f}s inactivity -> IDLE")
                stop_recording.set()
                sleep(0.3)
                _cache.unload_all()
                set_state(State.IDLE)

            sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        stop_recording.set()
        _cache.unload_all()
        try:
            board.cleanup()
        except Exception as e:
            print(f"[WARN] board.cleanup() failed: {e}")


if __name__ == "__main__":
    main()