"""
PC Translator Test (Windows)
Vosk + CTranslate2 OPUS EN<->ZH + pyttsx3
DUAL-MODEL VERSION:
- EN→ZH uses opus-en-zh-ct2
- ZH→EN uses opus-zh-en-ct2
"""

from __future__ import annotations
import os
import sys
import json
import wave
import time
from dataclasses import dataclass
from typing import Optional, List

import pyttsx3
import vosk
import ctranslate2
import sentencepiece as spm


# ======================================================
# Windows UTF-8 fix
# ======================================================

if os.name == "nt":
    os.system("chcp 65001 >nul")

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

CT2_EN_ZH_DIR = os.path.join(MODELS_DIR, "opus-en-zh-ct2")
CT2_ZH_EN_DIR = os.path.join(MODELS_DIR, "opus-zh-en-ct2")


# ======================================================
# SETTINGS
# ======================================================

CT2_COMPUTE_TYPE = "int8"
CT2_BEAM_SIZE = 2
CT2_MAX_DECODING_LEN = 128
CT2_NO_REPEAT_NGRAM = 3
CT2_REPETITION_PENALTY = 1.12

ENABLE_TTS = True


# ======================================================
# Helpers
# ======================================================

def require_dir(path: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(path)

def require_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

def maybe_append_eos(spx: spm.SentencePieceProcessor, toks: List[str]):
    if spx.piece_to_id("</s>") != spx.unk_id():
        return toks + ["</s>"]
    return toks

def strip_special(toks: List[str]):
    return [t for t in toks if t not in {"</s>", "<s>", "<pad>"}]


# ======================================================
# App State
# ======================================================

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
    tts: Optional[pyttsx3.Engine]


# ======================================================
# Init
# ======================================================

def load_ct2_model(model_dir: str) -> MTModel:
    require_dir(model_dir)
    require_file(os.path.join(model_dir, "model.bin"))
    require_file(os.path.join(model_dir, "source.spm"))
    require_file(os.path.join(model_dir, "target.spm"))

    translator = ctranslate2.Translator(
        model_dir,
        compute_type=CT2_COMPUTE_TYPE,
    )

    sp_src = spm.SentencePieceProcessor()
    sp_src.load(os.path.join(model_dir, "source.spm"))

    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.load(os.path.join(model_dir, "target.spm"))

    return MTModel(translator, sp_src, sp_tgt)


def init_app() -> AppState:
    print("Loading Vosk...")
    vosk_en = vosk.Model(VOSK_EN_DIR)
    vosk_zh = vosk.Model(VOSK_ZH_DIR)

    print("Loading EN→ZH model...")
    en_zh = load_ct2_model(CT2_EN_ZH_DIR)

    print("Loading ZH→EN model...")
    zh_en = load_ct2_model(CT2_ZH_EN_DIR)

    engine = pyttsx3.init() if ENABLE_TTS else None

    print("Ready.\n")

    return AppState(vosk_en, vosk_zh, en_zh, zh_en, engine)


# ======================================================
# STT
# ======================================================

def transcribe_wav(model: vosk.Model, wav_path: str) -> str:
    with wave.open(wav_path, "rb") as wf:
        rec = vosk.KaldiRecognizer(model, wf.getframerate())
        parts = []

        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                if res.get("text"):
                    parts.append(res["text"])

        final = json.loads(rec.FinalResult())
        if final.get("text"):
            parts.append(final["text"])

        return " ".join(parts).strip()


# ======================================================
# Translation
# ======================================================

def translate(mt: MTModel, text: str) -> str:
    if not text.strip():
        return ""

    toks = mt.sp_src.encode(text, out_type=str)
    toks = maybe_append_eos(mt.sp_src, toks)

    result = mt.translator.translate_batch(
        [toks],
        beam_size=CT2_BEAM_SIZE,
        max_decoding_length=CT2_MAX_DECODING_LEN,
        no_repeat_ngram_size=CT2_NO_REPEAT_NGRAM,
        repetition_penalty=CT2_REPETITION_PENALTY,
    )

    out_tokens = strip_special(result[0].hypotheses[0])
    return mt.sp_tgt.decode(out_tokens).strip()


def speak(engine, text):
    if engine and text:
        engine.say(text)
        engine.runAndWait()


# ======================================================
# CLI
# ======================================================

# ======================================================
# CLI
# ======================================================

def run_typed(app: AppState):
    direction = input("1=EN→ZH, 2=ZH→EN: ").strip()
    text = input("Text: ").strip()

    mt = app.en_zh if direction != "2" else app.zh_en
    label = "EN→ZH" if direction != "2" else "ZH→EN"

    t0 = time.time()
    out = translate(mt, text)

    print(f"\n[{label}] {text}")
    print("→", out)
    print(f"[time] {time.time()-t0:.3f}s")

    speak(app.tts, out)


def run_wav(app: AppState):
    lang = input("1=English speech, 2=Chinese speech: ").strip()

    if lang != "2":
        vosk_model = app.vosk_en
        mt = app.en_zh
        label = "EN→ZH"
    else:
        vosk_model = app.vosk_zh
        mt = app.zh_en
        label = "ZH→EN"

    text = transcribe_wav(vosk_model, REC_FILE)
    print("Recognized:", text)

    if not text:
        return

    t0 = time.time()
    out = translate(mt, text)

    print(f"\n[{label}] {text}")
    print("→", out)
    print(f"[time] {time.time()-t0:.3f}s")

    speak(app.tts, out)


# ======================================================
# Quick Test Samples (Option 3)
# ======================================================

def run_samples(app: AppState):
    samples = [
        ("EN→ZH", "I am allergic to peanuts."),
        ("EN→ZH", "Where is the nearest bathroom?"),
        ("EN→ZH", "Please speak slowly."),
        ("ZH→EN", "你好，我想点一杯咖啡。"),
        ("ZH→EN", "请问最近的地铁站在哪里？"),
    ]

    for label, text in samples:
        mt = app.en_zh if label == "EN→ZH" else app.zh_en

        t0 = time.time()
        out = translate(mt, text)
        dt = time.time() - t0

        print(f"\n[{label}] {text}")
        print("→", out)
        print(f"[time] {dt:.3f}s")

        speak(app.tts, out)


def main():
    app = init_app()

    while True:
        print("\n==============================")
        print("PC Translator Test (dual model)")
        print("==============================")
        print("1) Type text → Translate → Speak")
        print("2) WAV file → STT → Translate → Speak")
        print("3) Quick test samples")
        print("0) Quit")

        c = input("Choose: ").strip()

        if c == "0":
            break
        elif c == "1":
            run_typed(app)
        elif c == "2":
            run_wav(app)
        elif c == "3":
            run_samples(app)
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()