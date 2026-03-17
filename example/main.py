#!/usr/bin/env python3
"""
WhisPlay HAT Translator — EN <-> ZH
Vosk STT + CTranslate2 OPUS MT + Piper TTS
WhisPlay button logic:
  - First press:
      short press  (<1s) -> start Chinese first (ZH->EN)
      long press   (>=1s) -> start English first (EN->ZH)
  - While listening:
      next button press -> stop recording -> transcribe -> translate -> speak
      then automatically start listening for the other speaker
  - If 10 seconds of silence -> stop and return to IDLE

Python deps:
    pip install vosk ctranslate2 sentencepiece numpy pygame

System deps:
    sudo apt update
    sudo apt install -y alsa-utils

Piper:
    Use the standalone binary, e.g. ~/piper/piper
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import wave
import threading
import subprocess
from dataclasses import dataclass
from enum import Enum
from time import sleep
from typing import List, Optional

import numpy as np
import vosk
import ctranslate2
import sentencepiece as spm
import pygame

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARN] Pillow not installed — LCD image loading disabled")

try:
    sys.path.append(os.path.abspath("../Driver"))
    from WhisPlay import WhisPlayBoard
    WHISPLAY_AVAILABLE = True
except ImportError:
    WHISPLAY_AVAILABLE = False
    print("[WARN] WhisPlay driver not found — running headless")


# ─────────────────────────────────────────────────────────────────────────────
# State Machine
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
TTS_FILE = os.path.join(DATA_DIR, "tts_out.wav")

VOSK_EN_DIR = os.path.join(MODELS_DIR, "vosk-en")
VOSK_ZH_DIR = os.path.join(MODELS_DIR, "vosk-cn")

CT2_EN_ZH_DIR = os.path.join(MODELS_DIR, "opus-mt-en-zh-ctranslate2")
CT2_ZH_EN_DIR = os.path.join(MODELS_DIR, "opus-mt-zh-en-ctranslate2")

PIPER_BINARY = os.path.expanduser("~/piper/piper")
VOICE_EN = os.path.join(VOICES_DIR, "en_US-lessac-medium.onnx")
VOICE_ZH = os.path.join(VOICES_DIR, "zh_CN-huayan-x-low.onnx")


# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
CT2_COMPUTE_TYPE = "int8"
CT2_BEAM_SIZE = 2
CT2_MAX_DECODING_LEN = 256
CT2_NO_REPEAT_NGRAM = 3
CT2_REPETITION_PENALTY = 1.15

MIC_SAMPLE_RATE = 16000
MIC_CHANNELS = 1
MIC_BLOCK_SEC = 0.25
MIC_BYTES_PER_BLOCK = int(MIC_SAMPLE_RATE * MIC_BLOCK_SEC * 2 * MIC_CHANNELS)

SILENCE_RMS = 300
HOLD_DURATION = 1.0
SILENCE_TIMEOUT = 10.0

ENABLE_TTS = True


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MTModel:
    translator: ctranslate2.Translator
    sp_src: spm.SentencePieceProcessor
    sp_tgt: spm.SentencePieceProcessor


@dataclass
class AppState:
    vosk_en: vosk.Model
    vosk_zh: vosk.Model
    en_zh: MTModel
    zh_en: MTModel
    piper_ok: bool
    voice_en: str
    voice_zh: str
    card_name: str


# ─────────────────────────────────────────────────────────────────────────────
# Globals for board / app state
# ─────────────────────────────────────────────────────────────────────────────
board = None
images = {}

state = State.IDLE
eng_to_cn = True
press_time = 0.0

record_stop_event = threading.Event()
record_thread = None

lock = threading.RLock()


# ─────────────────────────────────────────────────────────────────────────────
# wm8960 helpers
# ─────────────────────────────────────────────────────────────────────────────
def find_wm8960_card_name() -> str:
    return "wm8960soundcard"


def find_wm8960_hw() -> str:
    card_name = find_wm8960_card_name()
    return f"hw:{card_name},0"


def set_wm8960_volume_stable(volume_level: str):
    card_name = find_wm8960_card_name()
    control_name = "Speaker"
    device_arg = f"hw:{card_name}"

    command = [
        "amixer",
        "-D", device_arg,
        "sset",
        control_name,
        volume_level,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"INFO: Set '{control_name}' volume to {volume_level} on '{card_name}'")
    except subprocess.CalledProcessError as e:
        print("ERROR: Failed to run amixer", file=sys.stderr)
        print(f"Command: {' '.join(command)}", file=sys.stderr)
        print(f"Return Code: {e.returncode}", file=sys.stderr)
        print(f"Error Output:\n{e.stderr}", file=sys.stderr)
    except FileNotFoundError:
        print("ERROR: amixer not found. Install alsa-utils.", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_image_as_rgb565(filepath, screen_width, screen_height):
    if not PIL_AVAILABLE:
        return None

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


def update_display(current_state: State):
    global board, images

    if board is None:
        return

    key = current_state.name.lower()
    img = images.get(key)
    if img is not None:
        board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img)


def set_state(new_state: State):
    global state
    with lock:
        state = new_state
        label = {
            State.IDLE: "IDLE",
            State.LISTENING: "LISTENING",
            State.PROCESSING: "PROCESSING",
        }[new_state]
        print(f"\n[STATE] {label}")
        update_display(new_state)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def _load_ct2(model_dir: str) -> MTModel:
    for f in ("model.bin", "source.spm", "target.spm"):
        p = os.path.join(model_dir, f)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing model file: {p}")

    translator = ctranslate2.Translator(
        model_dir,
        compute_type=CT2_COMPUTE_TYPE,
        inter_threads=1,
        intra_threads=2,
    )

    sp_src = spm.SentencePieceProcessor()
    sp_src.load(os.path.join(model_dir, "source.spm"))

    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.load(os.path.join(model_dir, "target.spm"))

    return MTModel(translator, sp_src, sp_tgt)


def _check_piper() -> bool:
    if not os.path.isfile(PIPER_BINARY) or not os.access(PIPER_BINARY, os.X_OK):
        print(f"[Piper] binary not found or not executable: {PIPER_BINARY}")
        return False

    missing = [v for v in (VOICE_EN, VOICE_ZH) if not os.path.isfile(v)]
    if missing:
        print(f"[Piper] missing voice files: {missing}")
        return False

    return True


def init_audio():
    pygame.mixer.init()
    try:
        set_wm8960_volume_stable("121")
    except Exception:
        pass


def init_app() -> AppState:
    vosk.SetLogLevel(-1)

    print("Loading Vosk EN ...")
    vosk_en = vosk.Model(VOSK_EN_DIR)

    print("Loading Vosk ZH ...")
    vosk_zh = vosk.Model(VOSK_ZH_DIR)

    print("Loading EN->ZH model ...")
    en_zh = _load_ct2(CT2_EN_ZH_DIR)

    print("Loading ZH->EN model ...")
    zh_en = _load_ct2(CT2_ZH_EN_DIR)

    piper_ok = _check_piper() if ENABLE_TTS else False
    if piper_ok:
        print("[Piper] OK")
    else:
        print("[Piper] disabled")

    return AppState(
        vosk_en=vosk_en,
        vosk_zh=vosk_zh,
        en_zh=en_zh,
        zh_en=zh_en,
        piper_ok=piper_ok,
        voice_en=VOICE_EN,
        voice_zh=VOICE_ZH,
        card_name=find_wm8960_card_name(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.abs(audio.astype(np.float32)).max()
    if peak < 1:
        return audio
    scale = (32767 * 0.9) / peak
    if scale <= 1.0:
        return audio
    return np.clip(audio.astype(np.float32) * scale, -32768, 32767).astype(np.int16)


def _pcm_to_wav_file(raw_pcm: bytes, out_path: str, sample_rate: int):
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(MIC_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_pcm)


# ─────────────────────────────────────────────────────────────────────────────
# Recording via ALSA arecord stream
# ─────────────────────────────────────────────────────────────────────────────
def record_until_button_or_silence(stop_event: threading.Event, out_path: str = REC_FILE) -> Optional[str]:
    """
    Stream audio from arecord stdout.
    Stop when:
      - stop_event is set by button press
      - 10 seconds of silence occur
    Returns:
      WAV path if stopped by button and speech was captured
      None if silence timeout -> return to idle
    """
    hw = find_wm8960_hw()

    cmd = [
        "arecord",
        "-D", hw,
        "-f", "S16_LE",
        "-r", str(MIC_SAMPLE_RATE),
        "-c", str(MIC_CHANNELS),
        "-t", "raw",
        "-q",
    ]

    print(f"[MIC] Recording from {hw} ...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    frames = []
    last_non_silent = time.time()

    try:
        while not stop_event.is_set():
            chunk = proc.stdout.read(MIC_BYTES_PER_BLOCK)
            if not chunk:
                break

            frames.append(chunk)

            block = np.frombuffer(chunk, dtype=np.int16)
            if block.size > 0:
                rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
                if rms > SILENCE_RMS:
                    last_non_silent = time.time()

            if time.time() - last_non_silent >= SILENCE_TIMEOUT:
                print(f"[MIC] {SILENCE_TIMEOUT:.0f}s silence detected -> returning to IDLE")
                stop_event.set()
                try:
                    proc.terminate()
                except Exception:
                    pass
                proc.wait(timeout=2)
                return None

        try:
            proc.terminate()
        except Exception:
            pass

        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if proc.stderr:
                proc.stderr.close()
        except Exception:
            pass

    raw_audio = b"".join(frames)
    if not raw_audio:
        print("[MIC] No audio captured")
        return None

    _pcm_to_wav_file(raw_audio, out_path, MIC_SAMPLE_RATE)
    print("[MIC] Recording stopped")
    return out_path


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
# TTS via Piper binary -> WAV -> pygame
# ─────────────────────────────────────────────────────────────────────────────
def _infer_piper_sample_rate(voice_path: str) -> int:
    json_path = voice_path + ".json"
    sample_rate = 22050

    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            sample_rate = cfg.get("audio", {}).get("sample_rate", sample_rate)
        except Exception:
            pass

    return sample_rate


def _run_piper_to_wav(voice_path: str, text: str, wav_path: str):
    sample_rate = _infer_piper_sample_rate(voice_path)
    json_path = voice_path + ".json"

    cmd = [PIPER_BINARY, "--model", voice_path, "--output_raw"]
    if os.path.isfile(json_path):
        cmd += ["--config", json_path]

    result = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace"))

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(result.stdout)


def speak(app: AppState, text: str, lang: str):
    if not ENABLE_TTS or not text or not app.piper_ok:
        return

    voice_path = app.voice_zh if lang == "zh" else app.voice_en

    try:
        _run_piper_to_wav(voice_path, text, TTS_FILE)
        pygame.mixer.music.load(TTS_FILE)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            sleep(0.05)

    except Exception as e:
        print(f"[TTS] Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Conversation flow
# ─────────────────────────────────────────────────────────────────────────────
def process_turn(app: AppState):
    global eng_to_cn

    set_state(State.PROCESSING)

    vosk_model = app.vosk_en if eng_to_cn else app.vosk_zh
    mt = app.en_zh if eng_to_cn else app.zh_en
    tgt_lang = "zh" if eng_to_cn else "en"
    direction = "EN->ZH" if eng_to_cn else "ZH->EN"

    raw = _transcribe_wav(vosk_model, REC_FILE)
    text = _dedup_stt(raw)

    print(f"Recognised: {text}" + (f"  (raw: {raw})" if raw != text else ""))

    if not text:
        print("[PROCESS] Nothing recognised -> continuing same direction")
        start_listening(app, keep_direction=True)
        return

    t0 = time.time()
    out = translate(mt, text)
    print(f"[{direction}] {text}\n-> {out}\n[{time.time() - t0:.2f}s]")

    speak(app, out, tgt_lang)

    eng_to_cn = not eng_to_cn
    start_listening(app, keep_direction=True)


def recording_worker(app: AppState):
    wav_path = record_until_button_or_silence(record_stop_event, REC_FILE)

    with lock:
        if wav_path is None:
            set_state(State.IDLE)
            return

    process_turn(app)


def start_listening(app: AppState, keep_direction: bool = True):
    global record_thread

    if not keep_direction:
        pass

    record_stop_event.clear()
    set_state(State.LISTENING)

    record_thread = threading.Thread(target=recording_worker, args=(app,), daemon=True)
    record_thread.start()


# ─────────────────────────────────────────────────────────────────────────────
# Button callbacks
# ─────────────────────────────────────────────────────────────────────────────
def on_press():
    global press_time
    press_time = time.time()


def on_release_factory(app: AppState):
    def on_release():
        global eng_to_cn

        duration = time.time() - press_time

        with lock:
            if state == State.IDLE:
                if duration >= HOLD_DURATION:
                    eng_to_cn = True
                    print("[BTN] Long press -> start EN->ZH")
                else:
                    eng_to_cn = False
                    print("[BTN] Short press -> start ZH->EN")

                start_listening(app, keep_direction=True)
                return

            if state == State.LISTENING:
                print("[BTN] Stop recording")
                record_stop_event.set()
                return

            if state == State.PROCESSING:
                print("[BTN] Ignored during processing")

    return on_release


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global board, images

    init_audio()
    app = init_app()

    if WHISPLAY_AVAILABLE:
        board = WhisPlayBoard()
        board.set_backlight(50)

        idle_img = os.path.join(IMGS_DIR, "passive.jpg")
        listening_img = os.path.join(IMGS_DIR, "recording.jpg")
        processing_img = os.path.join(IMGS_DIR, "playing.jpg")

        try:
            if os.path.isfile(idle_img):
                images["idle"] = load_image_as_rgb565(idle_img, board.LCD_WIDTH, board.LCD_HEIGHT)
            if os.path.isfile(listening_img):
                images["listening"] = load_image_as_rgb565(listening_img, board.LCD_WIDTH, board.LCD_HEIGHT)
            if os.path.isfile(processing_img):
                images["processing"] = load_image_as_rgb565(processing_img, board.LCD_WIDTH, board.LCD_HEIGHT)
        except Exception as e:
            print(f"[Display] Image load error: {e}")

        board.on_button_press(on_press)
        board.on_button_release(on_release_factory(app))

    set_state(State.IDLE)

    print("WhisPlay Translator ready")
    print("  Long first press  -> English first")
    print("  Short first press -> Chinese first")
    print("  Next press while listening -> stop, translate, speak")
    print(f"  {SILENCE_TIMEOUT:.0f}s silence -> return to idle")

    try:
        while True:
            sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        try:
            record_stop_event.set()
        except Exception:
            pass

        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

        try:
            pygame.mixer.quit()
        except Exception:
            pass

        if board is not None:
            try:
                board.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    main()