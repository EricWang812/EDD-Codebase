"""
PC Translator — EN <-> TH
Vosk STT  +  CTranslate2 HPLT MT  +  PyThaiTTS (Thai) / pyttsx3+espeak-ng (English)

Laptop / desktop adaptation of the WhisPlay HAT translator.
No hardware board required. Runs as an interactive CLI.

═══════════════════════════════════════════════════════════════
 SYSTEM DEPENDENCIES (install once)
═══════════════════════════════════════════════════════════════
    # Linux
    sudo apt install espeak-ng libportaudio2 portaudio19-dev

    # macOS
    brew install espeak portaudio

    # Windows — install eSpeak-NG from https://github.com/espeak-ng/espeak-ng/releases
    #           then add it to PATH

 PYTHON DEPENDENCIES
═══════════════════════════════════════════════════════════════
    pip install vosk ctranslate2 sentencepiece sounddevice numpy \
                pyttsx3 pythaitts pythainlp huggingface_hub

═══════════════════════════════════════════════════════════════
 MODEL SETUP (run once, requires internet)
═══════════════════════════════════════════════════════════════

 1. Download Vosk English model
    ────────────────────────────
    mkdir -p models/vosk-en
    wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    unzip vosk-model-small-en-us-0.15.zip -d models/vosk-en --strip-components=1

 2. Download Vosk Thai model
    ─────────────────────────
    mkdir -p models/vosk-th
    wget https://github.com/vistec-AI/commonvoice-th/releases/download/vosk-v1/model.zip
    unzip model.zip -d models/vosk-th --strip-components=1

 3. Download + convert HPLT Marian models
    ────────────────────────────────────────
    python3 -c "
    from huggingface_hub import snapshot_download
    snapshot_download('HPLT/translate-th-en-v2.0-hplt', local_dir='models/hplt-th-en')
    snapshot_download('HPLT/translate-en-th-v2.0-hplt', local_dir='models/hplt-en-th')
    "

    python3 -c "
    import ctranslate2
    ctranslate2.converters.MarianConverter(
        model_path='models/hplt-th-en/model.npz.best-chrf.npz',
        vocab_paths=['models/hplt-th-en/model.th-en.vocab'],
    ).convert('models/hplt-th-en-ct2', quantization='int8', force=True)
    ctranslate2.converters.MarianConverter(
        model_path='models/hplt-en-th/model.npz.best-chrf.npz',
        vocab_paths=['models/hplt-en-th/model.en-th.vocab'],
    ).convert('models/hplt-en-th-ct2', quantization='int8', force=True)
    print('Done.')
    "

 4. Pre-cache PyThaiTTS ONNX files (~4 files from HuggingFace)
    ─────────────────────────────────────────────────────────────
    python3 -c "from pythaitts import TTS; TTS(pretrained='lunarlist_onnx')"

═══════════════════════════════════════════════════════════════
 USAGE
═══════════════════════════════════════════════════════════════
    python3 translator_laptop.py              # normal mode
    python3 translator_laptop.py --no-audio   # skip all TTS/mic (text-only)
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

# ── sounddevice: requires PortAudio system library ────────────────────────────
try:
    import sounddevice as sd
    SD_AVAILABLE = True
except OSError as _sd_err:
    SD_AVAILABLE = False
    print(f"[WARN] sounddevice unavailable: {_sd_err}")
    print("[WARN] Fix (Linux): sudo apt install libportaudio2 portaudio19-dev")
    print("[WARN] Fix (macOS): brew install portaudio")
    print("[WARN] Live mic and audio playback disabled.\n")

import vosk
import ctranslate2
import sentencepiece as spm

# ── pyttsx3: English TTS — requires espeak-ng binary ─────────────────────────
try:
    import pyttsx3 as _pyttsx3_mod
    PYTTSX3_AVAILABLE = True
except ImportError:
    _pyttsx3_mod = None
    PYTTSX3_AVAILABLE = False
    print("[WARN] pyttsx3 not installed — run: pip install pyttsx3")

# ── PyThaiTTS: Thai speech synthesis ─────────────────────────────────────────
try:
    from pythaitts import TTS as ThaiTTS
    PYTHAITTS_AVAILABLE = True
except ImportError:
    ThaiTTS = None
    PYTHAITTS_AVAILABLE = False
    print("[WARN] pythaitts not installed — run: pip install pythaitts")

# ── PyThaiNLP: Thai word tokeniser (needed before MT) ────────────────────────
try:
    from pythainlp.tokenize import word_tokenize as th_word_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    th_word_tokenize = None
    PYTHAINLP_AVAILABLE = False
    print("[WARN] pythainlp not installed — run: pip install pythainlp")


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DATA_DIR   = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

REC_FILE     = os.path.join(DATA_DIR, "recorded_voice.wav")
TTS_OUT_FILE = os.path.join(DATA_DIR, "tts_out.wav")

VOSK_EN_DIR   = os.path.join(MODELS_DIR, "vosk-en")
VOSK_TH_DIR   = os.path.join(MODELS_DIR, "vosk-th")
CT2_EN_TH_DIR = os.path.join(MODELS_DIR, "hplt-en-th-ct2")
CT2_TH_EN_DIR = os.path.join(MODELS_DIR, "hplt-th-en-ct2")
SPM_EN_TH     = os.path.join(MODELS_DIR, "hplt-en-th", "model.en-th.spm")
SPM_TH_EN     = os.path.join(MODELS_DIR, "hplt-th-en", "model.th-en.spm")


# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
CT2_COMPUTE_TYPE       = "int8"
CT2_BEAM_SIZE          = 4
CT2_MAX_DECODING_LEN   = 256
CT2_NO_REPEAT_NGRAM    = 3
CT2_REPETITION_PENALTY = 1.15

MIC_SAMPLE_RATE = 16_000
MIC_CHANNELS    = 1
MIC_BLOCK_SEC   = 0.25
SILENCE_TIMEOUT = 2.5    # seconds of silence before auto-stop
MAX_RECORD_SEC  = 30
SILENCE_RMS     = 300    # RMS threshold below which a block is considered silent

ENABLE_TTS = True

# PyThaiTTS backend: "lunarlist_onnx" | "vachana" | "khanomtan" | "lunarlist"
THAI_TTS_MODEL   = "lunarlist_onnx"
# vachana speaker — only used when THAI_TTS_MODEL == "vachana"
THAI_TTS_SPEAKER = "th_f_1"

# pyttsx3 English speech rate (words per minute)
PYTTSX3_RATE = 160

# Characters supported by lunarlist_onnx's internal clean() function.
# Any character outside this set causes a KeyError crash inside the model.
_LUNARLIST_SAFE = frozenset(
    "กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลวศษสหฬอฮ"
    "ะัาำิีึืุูเแโใไๅ็่้๊๋์ "
)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MTModel:
    translator: ctranslate2.Translator
    sp:         spm.SentencePieceProcessor  # shared src+tgt SPM vocab


@dataclass
class AppState:
    vosk_en:      vosk.Model
    vosk_th:      vosk.Model
    en_th:        MTModel
    th_en:        MTModel
    thai_tts:     object           # ThaiTTS instance or None
    tts_engine:   object           # pyttsx3.Engine or None
    tts_voice_en: Optional[str]    # pyttsx3 English voice ID


# ─────────────────────────────────────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────────────────────────────────────
def _load_ct2(model_dir: str, spm_path: str) -> MTModel:
    """
    Load a CTranslate2 Marian model + its SentencePiece vocab.

    model_dir must be the output of ctranslate2.converters.MarianConverter.convert()
    spm_path is the .spm runtime vocab (distinct from the .vocab YAML used for conversion).
    """
    if not os.path.isfile(os.path.join(model_dir, "model.bin")):
        raise FileNotFoundError(
            f"CTranslate2 model.bin not found in: {model_dir}\n"
            "Run the MarianConverter conversion step in the module docstring."
        )
    if not os.path.isfile(spm_path):
        raise FileNotFoundError(
            f"SentencePiece vocab not found: {spm_path}\n"
            "Download the HPLT model directory from HuggingFace."
        )
    translator = ctranslate2.Translator(model_dir, compute_type=CT2_COMPUTE_TYPE)
    sp = spm.SentencePieceProcessor()
    sp.load(spm_path)
    return MTModel(translator, sp)


def _load_thai_tts() -> Optional[object]:
    """
    Load PyThaiTTS model.
    lunarlist_onnx downloads 4 ONNX files from HuggingFace on first run;
    pre-cache while online: python3 -c "from pythaitts import TTS; TTS(pretrained='lunarlist_onnx')"
    """
    if not PYTHAITTS_AVAILABLE:
        print("[WARN] PyThaiTTS unavailable — Thai TTS disabled.")
        return None
    try:
        print(f"Loading PyThaiTTS ({THAI_TTS_MODEL}) ...")
        tts = ThaiTTS(pretrained=THAI_TTS_MODEL)
        print("PyThaiTTS loaded.")
        return tts
    except Exception as e:
        print(f"[WARN] PyThaiTTS failed: {e}")
        print("[WARN] Pre-cache models: python3 -c \"from pythaitts import TTS; TTS(pretrained='lunarlist_onnx')\"")
        return None


def _load_pyttsx3() -> tuple:
    """
    Initialise pyttsx3 for English TTS.
    Returns (engine, voice_id) or (None, None) if unavailable.
    pyttsx3.init() raises RuntimeError when espeak-ng is not installed.
    """
    if not PYTTSX3_AVAILABLE:
        return None, None
    try:
        engine = _pyttsx3_mod.init()
        engine.setProperty("rate", PYTTSX3_RATE)
        engine.setProperty("volume", 1.0)
        voices = engine.getProperty("voices")
        # Prefer a voice explicitly labelled English; fall back to first available
        voice_en = next(
            (v.id for v in voices if "en" in v.id.lower() or "english" in v.name.lower()),
            voices[0].id if voices else None,
        )
        print(f"pyttsx3 loaded — EN voice: {voice_en}")
        return engine, voice_en
    except RuntimeError as e:
        print(f"[WARN] pyttsx3 init failed: {e}")
        print("[WARN] Install espeak-ng: sudo apt install espeak-ng  (Linux)")
        print("[WARN]                    brew install espeak           (macOS)")
        return None, None
    except Exception as e:
        print(f"[WARN] pyttsx3 unexpected error: {e}")
        return None, None


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
    tts_engine, tts_voice_en = _load_pyttsx3()
    print("\nReady.\n")
    return AppState(vosk_en, vosk_th, en_th, th_en,
                    thai_tts, tts_engine, tts_voice_en)


# ─────────────────────────────────────────────────────────────────────────────
# Audio pre-processing
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalise int16 audio to ~90% of full scale. Only amplifies; never clips loud input."""
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
    """Record from the default microphone until silence or max time."""
    if not SD_AVAILABLE:
        raise RuntimeError(
            "sounddevice is not available (PortAudio missing).\n"
            "Install: sudo apt install libportaudio2 portaudio19-dev  (Linux)\n"
            "         brew install portaudio                           (macOS)"
        )
    print(f"Listening ... (silence for {SILENCE_TIMEOUT:.0f}s stops recording)")
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
    # Collapse repeated Thai substrings (MT repetition artefact).
    # Thai Unicode block: U+0E00–U+0E7F
    text = re.sub(r'([\u0e00-\u0e7f]{2,6})\1{2,}', r'\1', text)
    return " ".join(text.split()).strip()


def _tokenize_thai_for_mt(text: str) -> str:
    """
    Insert spaces between Thai words before the SPM encoder.
    Thai has no word-boundary spaces natively; without this the encoder
    treats the whole utterance as one continuous token sequence.
    """
    if not text or not PYTHAINLP_AVAILABLE:
        return text
    tokens = th_word_tokenize(text, engine="newmm")
    return " ".join(t for t in tokens if t.strip())


def translate(mt: MTModel, text: str, src_lang: str) -> str:
    """
    Translate text using a CTranslate2 HPLT Marian model.
    ctranslate2.converters.MarianConverter sets add_source_eos=True internally;
    do NOT append </s> manually.
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


# ─────────────────────────────────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────────────────────────────────
def _sanitise_for_lunarlist(text: str) -> str:
    """
    Strip characters not in lunarlist_onnx's supported charset.
    The model's internal clean() uses dict_idx[char] with no KeyError guard —
    any unsupported character crashes synthesis.
    """
    return "".join(c for c in text if c in _LUNARLIST_SAFE).strip()


def _speak_thai(thai_tts, text: str):
    """Synthesise Thai via PyThaiTTS and play through sounddevice."""
    if thai_tts is None:
        print("  [TTS-TH] PyThaiTTS not loaded — skipping.")
        return

    # Preprocess manually so we can sanitise the result before synthesis
    try:
        from pythaitts.preprocess import preprocess_text
        preprocessed = preprocess_text(text)
    except Exception:
        preprocessed = text

    if THAI_TTS_MODEL == "lunarlist_onnx":
        preprocessed = _sanitise_for_lunarlist(preprocessed)

    if not preprocessed:
        print(f"  [TTS-TH] Nothing to speak after sanitisation (original: {text!r})")
        return

    print(f"  [TTS-TH] Synthesising: {preprocessed!r}")
    try:
        kwargs: dict = {
            "filename":   TTS_OUT_FILE,
            "preprocess": False,   # already preprocessed above
        }
        if THAI_TTS_MODEL == "vachana":
            kwargs["speaker_idx"] = THAI_TTS_SPEAKER

        thai_tts.tts(preprocessed, **kwargs)
        print(f"  [TTS-TH] WAV written: {os.path.getsize(TTS_OUT_FILE)} bytes")

        if not SD_AVAILABLE:
            print("  [TTS-TH] (audio playback skipped — sounddevice unavailable)")
            return

        with wave.open(TTS_OUT_FILE, "rb") as wf:
            rate = wf.getframerate()
            data = wf.readframes(wf.getnframes())
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio, samplerate=rate)
        sd.wait()

    except Exception as e:
        import traceback
        print(f"  [TTS-TH] Error: {e}")
        traceback.print_exc()


def _speak_english(app: AppState, text: str):
    """Synthesise English via pyttsx3 (uses espeak-ng on Linux/macOS)."""
    if app.tts_engine is None:
        print("  [TTS-EN] pyttsx3 not available — skipping.")
        return
    print(f"  [TTS-EN] Synthesising: {text!r}")
    try:
        if app.tts_voice_en:
            app.tts_engine.setProperty("voice", app.tts_voice_en)
        app.tts_engine.say(text)
        app.tts_engine.runAndWait()
    except Exception as e:
        import traceback
        print(f"  [TTS-EN] Error: {e}")
        traceback.print_exc()


def speak(app: AppState, text: str, lang: str):
    """Dispatch TTS to the correct engine based on target language."""
    if not ENABLE_TTS or not text:
        return
    print(f"  [TTS] lang={lang}")
    if lang == "th":
        _speak_thai(app.thai_tts, text)
    else:
        _speak_english(app, text)


# ─────────────────────────────────────────────────────────────────────────────
# CLI modes
# ─────────────────────────────────────────────────────────────────────────────
def run_typed(app: AppState):
    """Mode 1 — type text, translate, speak."""
    choice    = input("Direction — 1=EN->TH  2=TH->EN: ").strip()
    en_to_th  = choice != "2"
    mt        = app.en_th  if en_to_th else app.th_en
    label     = "EN->TH"   if en_to_th else "TH->EN"
    src_lang  = "en"       if en_to_th else "th"
    tgt_lang  = "th"       if en_to_th else "en"

    text = input("Text: ").strip()
    if not text:
        print("No input.")
        return
    t0  = time.time()
    out = translate(mt, text, src_lang)
    print(f"\n[{label}]  {text}\n  ->  {out}  [{time.time()-t0:.2f}s]\n")
    speak(app, out, tgt_lang)


def run_wav(app: AppState):
    """Mode 2 — transcribe an existing WAV file, translate, speak."""
    if not os.path.isfile(REC_FILE):
        print(f"No WAV file found at: {REC_FILE}")
        print("Record one first with Mode 4 (live mic), or place a file there manually.")
        return
    choice    = input("Speech language in WAV — 1=English  2=Thai: ").strip()
    en_input  = choice != "2"
    mt        = app.en_th    if en_input else app.th_en
    label     = "EN->TH"     if en_input else "TH->EN"
    src_lang  = "en"         if en_input else "th"
    tgt_lang  = "th"         if en_input else "en"
    vosk_mdl  = app.vosk_en  if en_input else app.vosk_th

    raw  = _transcribe_wav(vosk_mdl, REC_FILE)
    text = _dedup_stt(raw)
    print(f"Recognised: {text!r}" + (f"  (raw: {raw!r})" if raw != text else ""))
    if not text:
        print("Nothing recognised.")
        return
    t0  = time.time()
    out = translate(mt, text, src_lang)
    print(f"\n[{label}]  {text}\n  ->  {out}  [{time.time()-t0:.2f}s]\n")
    speak(app, out, tgt_lang)


def run_samples(app: AppState):
    """Mode 3 — run a set of built-in sample phrases through the full pipeline."""
    samples = [
        ("EN->TH", "en", "Hello, how are you today?"),
        ("EN->TH", "en", "I am allergic to peanuts."),
        ("EN->TH", "en", "Where is the nearest hospital?"),
        ("EN->TH", "en", "Please call an ambulance."),
        ("EN->TH", "en", "How much does this cost?"),
        ("TH->EN", "th", "สวัสดีครับ ยินดีที่ได้รู้จัก"),
        ("TH->EN", "th", "ขอโทษครับ คุณพูดภาษาอังกฤษได้ไหม"),
        ("TH->EN", "th", "ราคาเท่าไหร่ครับ"),
        ("TH->EN", "th", "ห้องน้ำอยู่ที่ไหนครับ"),
    ]
    for label, src_lang, text in samples:
        mt       = app.en_th if src_lang == "en" else app.th_en
        tgt_lang = "th"      if src_lang == "en" else "en"
        t0       = time.time()
        out      = translate(mt, text, src_lang)
        print(f"\n[{label}]  {text}\n  ->  {out}  [{time.time()-t0:.2f}s]")
        speak(app, out, tgt_lang)


def run_live_mic(app: AppState):
    """Mode 4 — record from microphone, transcribe, translate, speak."""
    if not SD_AVAILABLE:
        print("Live mic unavailable — sounddevice/PortAudio not installed.")
        return
    choice    = input("Speak in — 1=English  2=Thai: ").strip()
    en_input  = choice != "2"
    mt        = app.en_th    if en_input else app.th_en
    label     = "EN->TH"     if en_input else "TH->EN"
    src_lang  = "en"         if en_input else "th"
    tgt_lang  = "th"         if en_input else "en"
    vosk_mdl  = app.vosk_en  if en_input else app.vosk_th

    wav_path = record_mic()
    t0       = time.time()
    raw      = _transcribe_wav(vosk_mdl, wav_path)
    text     = _dedup_stt(raw)
    print(f"Recognised: {text!r}" + (f"  (raw: {raw!r})" if raw != text else ""))
    if not text:
        print("Nothing recognised — try speaking louder or closer to the mic.")
        return
    t1  = time.time()
    out = translate(mt, text, src_lang)
    print(f"\n[{label}]  {text}\n  ->  {out}")
    print(f"[STT {t1-t0:.2f}s | translate {time.time()-t1:.2f}s | total {time.time()-t0:.2f}s]\n")
    speak(app, out, tgt_lang)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    no_audio = "--no-audio" in sys.argv
    if no_audio:
        global ENABLE_TTS
        ENABLE_TTS = False
        print("[INFO] --no-audio flag set: TTS and mic disabled.\n")

    app = init_app()

    while True:
        print("══════════════════════════════════")
        print("   PC Translator  (EN <-> TH)    ")
        print("══════════════════════════════════")
        print("  1) Type text  -> translate -> speak")
        print("  2) WAV file   -> STT -> translate -> speak")
        print("  3) Sample phrases (built-in test set)")
        print("  4) Live mic   -> STT -> translate -> speak")
        print("  0) Quit")
        print("══════════════════════════════════")
        c = input("Choose: ").strip()

        if   c == "0": break
        elif c == "1": run_typed(app)
        elif c == "2": run_wav(app)
        elif c == "3": run_samples(app)
        elif c == "4": run_live_mic(app)
        else:          print("Invalid choice.\n")


if __name__ == "__main__":
    main()