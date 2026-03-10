"""
PC Translator (Windows + Pi) — EN <-> ZH
Vosk STT  +  CTranslate2 OPUS MT  +  Piper TTS (Python package)

Install deps once:
    pip install vosk ctranslate2 sentencepiece sounddevice numpy pyttsx3 piper-tts
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import wave
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pyttsx3
import sounddevice as sd
import vosk
import ctranslate2
import sentencepiece as spm

try:
    from piper.voice import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Windows UTF-8 fix
# ─────────────────────────────────────────────────────────────────────────────
if os.name == "nt":
    os.system("chcp 65001 >nul")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DATA_DIR   = os.path.join(HERE, "data")
VOICES_DIR = os.path.join(HERE, "voices")
os.makedirs(DATA_DIR, exist_ok=True)

REC_FILE      = os.path.join(DATA_DIR, "recorded_voice.wav")
VOSK_EN_DIR   = os.path.join(MODELS_DIR, "vosk-en")
VOSK_ZH_DIR   = os.path.join(MODELS_DIR, "vosk-cn")
CT2_EN_ZH_DIR = os.path.join(MODELS_DIR, "opus-en-zh-ct2")
CT2_ZH_EN_DIR = os.path.join(MODELS_DIR, "opus-zh-en-ct2")
VOICE_EN      = os.path.join(VOICES_DIR, "en_US-lessac-low.onnx")
VOICE_ZH      = os.path.join(VOICES_DIR, "zh_CN-huayan-x-low.onnx")

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
CT2_COMPUTE_TYPE       = "int8"
CT2_BEAM_SIZE          = 2
CT2_MAX_DECODING_LEN   = 256
CT2_NO_REPEAT_NGRAM    = 3
CT2_REPETITION_PENALTY = 1.15

MIC_SAMPLE_RATE = 16_000
MIC_CHANNELS    = 1
MIC_BLOCK_SEC   = 0.25
SILENCE_TIMEOUT = 2.5   # raised: avoids cutting off mid-sentence pauses
MAX_RECORD_SEC  = 30
SILENCE_RMS     = 300   # raised from 150: background hiss no longer trips end-of-speech

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
    vosk_en:      vosk.Model
    vosk_zh:      vosk.Model
    en_zh:        MTModel
    zh_en:        MTModel
    piper_en:     object
    piper_zh:     object
    tts_engine:   Optional[pyttsx3.Engine]
    tts_voice_en: Optional[str]
    tts_voice_zh: Optional[str]

# ─────────────────────────────────────────────────────────────────────────────
# Init
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
    else:
        print("piper-tts not installed — run: pip install piper-tts")

    tts_engine = tts_voice_en = tts_voice_zh = None
    if ENABLE_TTS:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", 160)
        tts_engine.setProperty("volume", 1.0)
        voices = tts_engine.getProperty("voices")
        tts_voice_en = voices[0].id if voices else None
        tts_voice_zh = next(
            (v.id for v in voices if "zh" in v.id.lower()
             or any(n in v.name.lower() for n in ("huihui", "yaoyao", "xiaoxiao"))),
            None,
        )
        print(f"pyttsx3 fallback — EN: {bool(tts_voice_en)}  ZH: {bool(tts_voice_zh)}")

    print("\nReady.\n")
    return AppState(vosk_en, vosk_zh, en_zh, zh_en,
                    piper_en, piper_zh, tts_engine, tts_voice_en, tts_voice_zh)

# ─────────────────────────────────────────────────────────────────────────────
# Audio pre-processing — pure numpy, zero model cost
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Peak-normalize int16 audio to ~90% of full scale.
    Only amplifies quiet recordings; leaves loud ones untouched to avoid clipping.
    """
    peak = np.abs(audio.astype(np.float32)).max()
    if peak < 1:
        return audio
    scale = (32767 * 0.9) / peak
    if scale < 1.0:
        return audio
    return np.clip(audio.astype(np.float32) * scale, -32768, 32767).astype(np.int16)

# ─────────────────────────────────────────────────────────────────────────────
# STT
# ─────────────────────────────────────────────────────────────────────────────
_FILLER_RE = re.compile(r'\b(uh+|um+|hmm+|huh|mm+|ah+|er+)\b', re.IGNORECASE)

def _strip_fillers(text: str) -> str:
    """Remove filler words that pollute the translation input."""
    return ' '.join(_FILLER_RE.sub('', text).split())


def _transcribe_wav(model: vosk.Model, wav_path: str) -> str:
    """
    Transcribe a WAV file with Vosk.

    Changes vs original (no compute cost increase):
    - Audio peak-normalised before decoding for consistent signal level.
    - Partial results harvested as fallback for trailing words Vosk doesn't flush.
    - Filler tokens (uh, um, hmm…) stripped from every segment.
    """
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

    while True:
        data = buf.read(8000)   # 4000 int16 samples
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
    """
    Remove duplicated phrases Vosk occasionally produces.

    Handles:
    1. Exact prefix repeat:          "hello hello world"   -> "hello world"
    2. Full-string repeat (recurse): "thank you thank you" -> "thank you"
    3. Consecutive identical tokens: "的 的 的 猫"         -> "的 猫"
    """
    words = text.split()
    n     = len(words)
    if n == 0:
        return text

    half = n // 2
    if half > 0 and words[:half] == words[half:half * 2]:
        return _dedup_stt(" ".join(words[half:]))   # recurse for tripling

    for size in range(half, 0, -1):
        if words[:size] == words[size:size * 2]:
            return " ".join(words[size:])

    deduped = [words[0]]
    for w in words[1:]:
        if w != deduped[-1]:
            deduped.append(w)
    return " ".join(deduped)


def record_mic(out_path: str = REC_FILE) -> str:
    print(f"Listening ... (silence for {SILENCE_TIMEOUT:.0f}s to stop)")
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
                started = True
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

    if hasattr(voice, "synthesize_wav"):
        out_wav = os.path.join(DATA_DIR, "tts_out.wav")
        with wave.open(out_wav, "wb") as wf:
            voice.synthesize_wav(text, wf)
        with wave.open(out_wav, "rb") as wf:
            rate = wf.getframerate()
            data = wf.readframes(wf.getnframes())
        sd.play(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0,
                samplerate=rate)
        sd.wait()
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

    raise RuntimeError("No supported synthesize method found on PiperVoice object.")


def speak(app: AppState, text: str, lang: str):
    if not ENABLE_TTS or not text:
        return
    voice = app.piper_zh if lang == "zh" else app.piper_en
    if voice:
        try:
            _piper_play(voice, text)
            return
        except Exception as e:
            print(f"  [TTS] Piper error: {e} — falling back to pyttsx3")
    if not app.tts_engine:
        print("  [TTS] No TTS available.")
        return
    vid = app.tts_voice_zh if (lang == "zh" and app.tts_voice_zh) else app.tts_voice_en
    if vid:
        app.tts_engine.setProperty("voice", vid)
    app.tts_engine.say(text)
    app.tts_engine.runAndWait()

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def run_typed(app: AppState):
    en_to_zh = input("Direction — 1=EN->ZH  2=ZH->EN: ").strip() != "2"
    mt, label, tgt = (app.en_zh, "EN->ZH", "zh") if en_to_zh else (app.zh_en, "ZH->EN", "en")
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
        print("Nothing recognised — try speaking louder.")
        return
    t1  = time.time()
    out = translate(mt, text)
    print(f"[{label}] {text}\n->  {out}")
    print(f"[translate {time.time()-t1:.2f}s | total {time.time()-t0:.2f}s]\n")
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


def main():
    app = init_app()
    while True:
        print("══════════════════════════════")
        print("  PC Translator  (EN <-> ZH) ")
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