"""
Pi 5 Translator — EN <-> ZH
Vosk STT  +  CTranslate2 OPUS MT  +  Piper TTS (ARM64 binary)

Install deps:
    pip install vosk ctranslate2 sentencepiece sounddevice numpy

Piper TTS (ARM64 binary — do NOT use pip install piper-tts on Pi):
    wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
    tar -xzf piper_linux_aarch64.tar.gz -C ~/piper
    # Then confirm PIPER_BINARY path below
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import sounddevice as sd
import vosk
import ctranslate2
import sentencepiece as spm

# ─────────────────────────────────────────────────────────────────────────────
# Paths — edit these to match your layout
# ─────────────────────────────────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DATA_DIR   = os.path.join(HERE, "data")
VOICES_DIR = os.path.join(HERE, "voices")
os.makedirs(DATA_DIR, exist_ok=True)

REC_FILE      = os.path.join(DATA_DIR, "recorded_voice.wav")

# Vosk — use full models on Pi 5 (not -small); much better accuracy
VOSK_EN_DIR   = os.path.join(MODELS_DIR, "vosk-model-en-us-0.22")
VOSK_ZH_DIR   = os.path.join(MODELS_DIR, "vosk-model-cn-0.22")

# CTranslate2 OPUS-MT models
CT2_EN_ZH_DIR = os.path.join(MODELS_DIR, "opus-en-zh-ct2")
CT2_ZH_EN_DIR = os.path.join(MODELS_DIR, "opus-zh-en-ct2")

# Piper TTS — ARM64 binary + ONNX voice files
# Download: https://github.com/rhasspy/piper/releases  (piper_linux_aarch64.tar.gz)
PIPER_BINARY  = os.path.expanduser("~/piper/piper")
VOICE_EN      = os.path.join(VOICES_DIR, "en_US-lessac-medium.onnx")   # medium quality fine on Pi 5
VOICE_ZH      = os.path.join(VOICES_DIR, "zh_CN-huayan-x-low.onnx")

# ─────────────────────────────────────────────────────────────────────────────
# Tuning — relaxed for Pi 5 (4-core A76, 4–8 GB RAM)
# ─────────────────────────────────────────────────────────────────────────────
CT2_COMPUTE_TYPE       = "int8"   # try "float32" if translation quality feels off
CT2_BEAM_SIZE          = 4        # was 2 — better quality, Pi 5 can handle it
CT2_MAX_DECODING_LEN   = 512      # was 256
CT2_NO_REPEAT_NGRAM    = 3
CT2_REPETITION_PENALTY = 1.15

MIC_SAMPLE_RATE = 16_000
MIC_CHANNELS    = 1
MIC_BLOCK_SEC   = 0.1             # was 0.25 — finer VAD on faster CPU
SILENCE_TIMEOUT = 2.0             # was 2.5 — snappier end-of-speech
MAX_RECORD_SEC  = 30
SILENCE_RMS     = 300

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
    piper_ok: bool
    voice_en: str
    voice_zh: str


# ─────────────────────────────────────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────────────────────────────────────
def _load_ct2(model_dir: str) -> MTModel:
    for f in ("model.bin", "source.spm", "target.spm"):
        p = os.path.join(model_dir, f)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing model file: {p}")
    translator = ctranslate2.Translator(
        model_dir,
        compute_type=CT2_COMPUTE_TYPE,
        inter_threads=2,   # Pi 5 has 4 cores; 2 threads per translator is efficient
        intra_threads=2,
    )
    sp_src = spm.SentencePieceProcessor()
    sp_src.load(os.path.join(model_dir, "source.spm"))
    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.load(os.path.join(model_dir, "target.spm"))
    return MTModel(translator, sp_src, sp_tgt)


def _check_piper() -> bool:
    """Return True only if the Piper binary is executable and voice files exist."""
    if not os.path.isfile(PIPER_BINARY) or not os.access(PIPER_BINARY, os.X_OK):
        print(f"  [Piper] binary not found / not executable: {PIPER_BINARY}")
        print("  Download: https://github.com/rhasspy/piper/releases  (piper_linux_aarch64.tar.gz)")
        return False
    missing = [v for v in (VOICE_EN, VOICE_ZH) if not os.path.isfile(v)]
    if missing:
        print(f"  [Piper] missing voice files: {missing}")
        return False
    print("  [Piper] binary + voices OK")
    return True


def init_app() -> AppState:
    vosk.SetLogLevel(-1)   # suppress noisy Vosk logs
    print("Loading Vosk EN ...")
    vosk_en = vosk.Model(VOSK_EN_DIR)
    print("Loading Vosk ZH ...")
    vosk_zh = vosk.Model(VOSK_ZH_DIR)
    print("Loading EN->ZH model ...")
    en_zh = _load_ct2(CT2_EN_ZH_DIR)
    print("Loading ZH->EN model ...")
    zh_en = _load_ct2(CT2_ZH_EN_DIR)

    piper_ok = False
    if ENABLE_TTS:
        print("Checking Piper TTS ...")
        piper_ok = _check_piper()
        if not piper_ok:
            print("  TTS disabled — install Piper ARM64 binary to enable spoken output.")

    print("\nReady.\n")
    return AppState(vosk_en, vosk_zh, en_zh, zh_en, piper_ok, VOICE_EN, VOICE_ZH)


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalise int16 to ~90 % full scale without clipping."""
    peak = np.abs(audio.astype(np.float32)).max()
    if peak < 1:
        return audio
    scale = (32767 * 0.9) / peak
    if scale <= 1.0:
        return audio
    return np.clip(audio.astype(np.float32) * scale, -32768, 32767).astype(np.int16)


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

    rec          = vosk.KaldiRecognizer(model, rate)
    rec.SetWords(True)
    parts        = []
    last_partial = ""

    # Larger read chunk on Pi 5 — 1 s worth of audio, reduces loop overhead
    CHUNK = MIC_SAMPLE_RATE * 2   # bytes (int16)
    while True:
        data = buf.read(CHUNK)
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
    """Remove duplicated phrases Vosk occasionally produces."""
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


def record_mic(out_path: str = REC_FILE) -> str:
    print(f"Listening ... (silence for {SILENCE_TIMEOUT:.1f}s to stop, max {MAX_RECORD_SEC}s)")
    frames       = []
    block_size   = int(MIC_SAMPLE_RATE * MIC_BLOCK_SEC)
    max_silent   = int(SILENCE_TIMEOUT / MIC_BLOCK_SEC)
    max_blocks   = int(MAX_RECORD_SEC  / MIC_BLOCK_SEC)
    silent_count = 0
    started      = False

    with sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=MIC_CHANNELS,
                        dtype="int16", blocksize=block_size) as stream:
        for _ in range(max_blocks):
            block, _ = stream.read(block_size)
            rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
            frames.append(block.copy())
            if rms > SILENCE_RMS:
                started      = True
                silent_count = 0
            elif started:
                silent_count += 1
                if silent_count >= max_silent:
                    break

    print("Recording stopped.")
    audio = np.concatenate(frames, axis=0)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(MIC_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(MIC_SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return out_path


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
# TTS — Piper via subprocess (ARM64 binary, no pip package needed)
# ─────────────────────────────────────────────────────────────────────────────
def speak(app: AppState, text: str, lang: str):
    """
    Synthesise speech with the Piper binary and play via sounddevice.

    Piper is invoked as:
        echo "text" | piper --model voice.onnx --output_raw
    Raw 16-bit PCM is captured and played with sounddevice (no aplay needed).
    """
    if not ENABLE_TTS or not text or not app.piper_ok:
        return

    voice_path = app.voice_zh if lang == "zh" else app.voice_en
    json_path  = voice_path + ".json"

    cmd = [PIPER_BINARY, "--model", voice_path, "--output_raw"]
    if os.path.isfile(json_path):
        cmd += ["--config", json_path]

    # Infer sample rate from voice config (Piper raw output has no WAV header)
    sample_rate = 22050   # safe default for en_US-lessac-medium
    if os.path.isfile(json_path):
        try:
            cfg = json.load(open(json_path))
            sample_rate = cfg.get("audio", {}).get("sample_rate", sample_rate)
        except Exception:
            pass

    try:
        result = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"  [TTS] Piper error: {result.stderr.decode(errors='replace')}")
            return

        audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio, samplerate=sample_rate)
        sd.wait()

    except subprocess.TimeoutExpired:
        print("  [TTS] Piper timed out.")
    except Exception as e:
        print(f"  [TTS] Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI modes
# ─────────────────────────────────────────────────────────────────────────────
def run_typed(app: AppState):
    en_to_zh = input("Direction — 1=EN->ZH  2=ZH->EN: ").strip() != "2"
    mt, label, tgt = (
        (app.en_zh, "EN->ZH", "zh") if en_to_zh
        else (app.zh_en, "ZH->EN", "en")
    )
    text = input("Text: ").strip()
    t0   = time.time()
    out  = translate(mt, text)
    print(f"\n[{label}] {text}\n->  {out}\n[{time.time()-t0:.2f}s]\n")
    speak(app, out, tgt)


def run_wav(app: AppState):
    en_input = input("Speech language — 1=English  2=Chinese: ").strip() != "2"
    vosk_model, mt, label, tgt = (
        (app.vosk_en, app.en_zh, "EN->ZH", "zh") if en_input
        else (app.vosk_zh, app.zh_en, "ZH->EN", "en")
    )
    if not os.path.isfile(REC_FILE):
        print(f"No WAV found at {REC_FILE}")
        return
    text = _transcribe_wav(vosk_model, REC_FILE)
    print(f"Recognised: {text}")
    if not text:
        return
    t0  = time.time()
    out = translate(mt, text)
    print(f"\n[{label}] {text}\n->  {out}\n[{time.time()-t0:.2f}s]\n")
    speak(app, out, tgt)


def run_live_mic(app: AppState):
    en_input = input("Speak in — 1=English  2=Chinese: ").strip() != "2"
    vosk_model, mt, label, tgt = (
        (app.vosk_en, app.en_zh, "EN->ZH", "zh") if en_input
        else (app.vosk_zh, app.zh_en, "ZH->EN", "en")
    )
    wav_path = record_mic()
    t0   = time.time()
    raw  = _transcribe_wav(vosk_model, wav_path)
    text = _dedup_stt(raw)
    print(f"Recognised: {text}" + (f"  (raw: {raw})" if raw != text else ""))
    if not text:
        print("Nothing recognised — try speaking louder or closer to the mic.")
        return
    t1  = time.time()
    out = translate(mt, text)
    print(f"[{label}] {text}\n->  {out}")
    print(f"[STT {t1-t0:.2f}s | translate {time.time()-t1:.2f}s | total {time.time()-t0:.2f}s]\n")
    speak(app, out, tgt)


def run_samples(app: AppState):
    samples = [
        ("EN->ZH", "I am allergic to peanuts."),
        ("EN->ZH", "Where is the nearest bathroom?"),
        ("EN->ZH", "Please call an ambulance."),
        ("ZH->EN", "你好，我想点一杯咖啡。"),
        ("ZH->EN", "请问最近的地铁站在哪里？"),
        ("ZH->EN", "我对花生过敏，请不要放花生。"),
    ]
    for label, text in samples:
        en_to_zh = label == "EN->ZH"
        mt  = app.en_zh if en_to_zh else app.zh_en
        tgt = "zh"      if en_to_zh else "en"
        t0  = time.time()
        out = translate(mt, text)
        print(f"\n[{label}] {text}\n->  {out}  [{time.time()-t0:.2f}s]")
        speak(app, out, tgt)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    app = init_app()
    while True:
        print("══════════════════════════════")
        print("  Pi 5 Translator (EN <-> ZH) ")
        print("══════════════════════════════")
        print("  1) Type text  -> translate -> speak")
        print("  2) WAV file   -> STT -> translate -> speak")
        print("  3) Quick test samples")
        print("  4) Live mic   -> STT -> translate -> speak")
        print("  0) Quit")
        print("══════════════════════════════")
        c = input("Choose: ").strip()
        if   c == "0": break
        elif c == "1": run_typed(app)
        elif c == "2": run_wav(app)
        elif c == "3": run_samples(app)
        elif c == "4": run_live_mic(app)
        else:          print("Invalid choice.")


if __name__ == "__main__":
    main()