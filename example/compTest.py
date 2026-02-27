"""
PC Translator Test (Windows) — Vosk + CTranslate2 OPUS EN<->ZH + pyttsx3
SINGLE-MODEL VERSION:
- Only uses models/opus-en-zh-ct2 for BOTH directions (so no opus-zh-en-ct2 needed)
- Windows UTF-8 console fix
- SentencePiece tokenizers (source.spm / target.spm)
- EOS + anti-repetition decoding

Folder layout:
example/
  compTest_pc.py
  models/
    vosk-en/
    vosk-cn/
    opus-en-zh-ct2/
      model.bin
      source.spm
      target.spm
      ...
  data/
    recorded_voice.wav
"""

from __future__ import annotations

import os
import sys
import json
import wave
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import pyttsx3
import vosk
import ctranslate2
import sentencepiece as spm


# ======================================================
# Windows console UTF-8 fix
# ======================================================

if os.name == "nt":
    try:
        os.system("chcp 65001 >nul")
    except Exception:
        pass

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ======================================================
# PATHS
# ======================================================

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(HERE, "models")
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

REC_FILE = os.path.join(DATA_DIR, "recorded_voice.wav")

VOSK_EN_DIR = os.path.join(MODELS_DIR, "vosk-en")
VOSK_ZH_DIR = os.path.join(MODELS_DIR, "vosk-cn")

# Single MT model used for BOTH directions:
CT2_DIR = os.path.join(MODELS_DIR, "opus-en-zh-ct2")


# ======================================================
# SETTINGS
# ======================================================

CT2_COMPUTE_TYPE = "int8"
CT2_INTER_THREADS = 1
CT2_INTRA_THREADS = 2

CT2_BEAM_SIZE = 2
CT2_MAX_DECODING_LEN = 128
CT2_NO_REPEAT_NGRAM = 3
CT2_REPETITION_PENALTY = 1.12

ENABLE_TTS = True


# ======================================================
# Helpers
# ======================================================

def require_dir(path: str, label: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing {label} directory:\n  {path}")

def require_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {label} file:\n  {path}")

def sp_has_piece(spx: spm.SentencePieceProcessor, piece: str) -> bool:
    try:
        return spx.piece_to_id(piece) != spx.unk_id()
    except Exception:
        return False

def maybe_append_eos(spx: spm.SentencePieceProcessor, toks: List[str]) -> List[str]:
    if sp_has_piece(spx, "</s>"):
        return toks + ["</s>"]
    return toks

def strip_special(toks: List[str]) -> List[str]:
    specials = {"</s>", "<s>", "<pad>"}
    return [t for t in toks if t not in specials]


# ======================================================
# App state
# ======================================================

@dataclass
class AppState:
    vosk_en: vosk.Model
    vosk_zh: vosk.Model
    translator: ctranslate2.Translator
    sp_src: spm.SentencePieceProcessor
    sp_tgt: spm.SentencePieceProcessor
    tts: Optional[pyttsx3.Engine]


# ======================================================
# Init
# ======================================================

def init_app() -> AppState:
    require_dir(VOSK_EN_DIR, "VOSK_EN_DIR")
    require_dir(VOSK_ZH_DIR, "VOSK_ZH_DIR")
    require_dir(CT2_DIR, "CT2_DIR")

    require_file(os.path.join(VOSK_EN_DIR, "conf", "model.conf"), "Vosk EN conf/model.conf")
    require_file(os.path.join(VOSK_ZH_DIR, "conf", "model.conf"), "Vosk ZH conf/model.conf")

    require_file(os.path.join(CT2_DIR, "model.bin"), "CT2 model.bin")
    require_file(os.path.join(CT2_DIR, "source.spm"), "CT2 source.spm")
    require_file(os.path.join(CT2_DIR, "target.spm"), "CT2 target.spm")

    print("Base dir:", HERE)
    print("Using SINGLE CT2 model:", CT2_DIR)

    print("\nLoading Vosk models...")
    vosk_en = vosk.Model(VOSK_EN_DIR)
    vosk_zh = vosk.Model(VOSK_ZH_DIR)

    print("\nLoading CTranslate2 model...")
    translator = ctranslate2.Translator(
        CT2_DIR,
        compute_type=CT2_COMPUTE_TYPE,
        inter_threads=CT2_INTER_THREADS,
        intra_threads=CT2_INTRA_THREADS,
    )

    print("\nLoading SentencePiece tokenizers...")
    sp_src = spm.SentencePieceProcessor()
    sp_src.load(os.path.join(CT2_DIR, "source.spm"))

    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.load(os.path.join(CT2_DIR, "target.spm"))

    engine = pyttsx3.init() if ENABLE_TTS else None

    print("\nReady.\n")
    return AppState(
        vosk_en=vosk_en,
        vosk_zh=vosk_zh,
        translator=translator,
        sp_src=sp_src,
        sp_tgt=sp_tgt,
        tts=engine,
    )


# ======================================================
# STT
# ======================================================

def transcribe_wav(vosk_model: vosk.Model, wav_path: str) -> str:
    require_file(wav_path, "WAV input (data/recorded_voice.wav)")

    with wave.open(wav_path, "rb") as wf:
        fr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()

        if ch != 1:
            print(f"[WARN] WAV channels={ch}. Recommend mono (1).")
        if fr != 16000:
            print(f"[WARN] WAV framerate={fr}. Recommend 16000 Hz.")
        if sw != 2:
            print(f"[WARN] WAV sampwidth={sw}. Recommend 16-bit PCM (2 bytes).")

        rec = vosk.KaldiRecognizer(vosk_model, fr)

        parts: List[str] = []
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                txt = res.get("text", "").strip()
                if txt:
                    parts.append(txt)

        final_res = json.loads(rec.FinalResult())
        txt = final_res.get("text", "").strip()
        if txt:
            parts.append(txt)

        return " ".join(parts).strip()


# ======================================================
# MT (single model)
# ======================================================

def translate_text(app: AppState, text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    toks = app.sp_src.encode(text, out_type=str)
    toks = maybe_append_eos(app.sp_src, toks)

    result = app.translator.translate_batch(
        [toks],
        beam_size=CT2_BEAM_SIZE,
        max_decoding_length=CT2_MAX_DECODING_LEN,
        no_repeat_ngram_size=CT2_NO_REPEAT_NGRAM,
        repetition_penalty=CT2_REPETITION_PENALTY,
    )

    out_toks = strip_special(result[0].hypotheses[0])
    out = app.sp_tgt.decode(out_toks).strip()
    out = " ".join(out.split())
    return out


def speak(engine: Optional[pyttsx3.Engine], text: str) -> None:
    if engine is None or not text.strip():
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[WARN] TTS failed: {e}")


# ======================================================
# CLI
# ======================================================

def menu() -> str:
    print("\n==============================")
    print("PC Translator Test (single model)")
    print("==============================")
    print("1) Type text → Translate → Speak")
    print("2) WAV file → STT → Translate → Speak")
    print("3) Quick test samples")
    print("0) Quit")
    return input("Choose: ").strip()


def run_typed(app: AppState) -> None:
    direction = input("Direction label only (1=EN→ZH, 2=ZH→EN): ").strip()
    label = "EN→ZH" if direction != "2" else "ZH→EN"

    text = input("Enter text: ").strip()

    t0 = time.time()
    out = translate_text(app, text)
    dt = time.time() - t0

    print(f"\n[{label}] {text}")
    print("→", out)
    print(f"[time] {dt:.3f}s")
    speak(app.tts, out)


def run_wav(app: AppState) -> None:
    print(f"\nWAV path:\n  {REC_FILE}")
    lang = input("WAV language (1=English speech, 2=Chinese speech): ").strip()
    english_speech = (lang != "2")

    vosk_model = app.vosk_en if english_speech else app.vosk_zh
    label = "EN→ZH" if english_speech else "ZH→EN"

    t0 = time.time()
    text = transcribe_wav(vosk_model, REC_FILE)
    dt_stt = time.time() - t0

    print("\nRecognized:", text if text else "(empty)")
    print(f"[stt time] {dt_stt:.3f}s")

    if not text:
        return

    t1 = time.time()
    out = translate_text(app, text)
    dt_tr = time.time() - t1

    print(f"\n[{label}] {text}")
    print("→", out)
    print(f"[translate time] {dt_tr:.3f}s")
    speak(app.tts, out)


def run_samples(app: AppState) -> None:
    samples = [
        ("EN→ZH", "I am allergic to peanuts."),
        ("EN→ZH", "Where is the nearest bathroom?"),
        ("EN→ZH", "Please speak slowly."),
        ("ZH→EN", "你好，我想点一杯咖啡。"),
        ("ZH→EN", "请问最近的地铁站在哪里？"),
    ]

    for label, s in samples:
        t0 = time.time()
        out = translate_text(app, s)
        dt = time.time() - t0
        print(f"\n[{label}] {s}")
        print("→", out)
        print(f"[time] {dt:.3f}s")
        speak(app.tts, out)


def main() -> None:
    app = init_app()
    while True:
        c = menu()
        if c == "0":
            break
        if c == "1":
            run_typed(app)
        elif c == "2":
            run_wav(app)
        elif c == "3":
            run_samples(app)
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()