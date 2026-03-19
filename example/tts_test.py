"""
Piper TTS interactive test — type text, hear it spoken.
Auto-detects language (English or Chinese) based on input.

Usage:
    python3 test_tts.py

Dependencies:
    pip install piper-tts sounddevice numpy
    sudo apt install espeak-ng   # Linux
    # macOS: brew install espeak-ng
"""

import io
import os
import sys
import wave

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("[ERROR] sounddevice not installed — run: pip install sounddevice")
    sys.exit(1)

try:
    from piper.voice import PiperVoice
except ImportError:
    print("[ERROR] piper-tts not installed — run: pip install piper-tts")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
HERE       = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(HERE, "voices")

VOICE_EN = os.path.join(VOICES_DIR, "en_US-lessac-low.onnx")
VOICE_ZH = os.path.join(VOICES_DIR, "zh_CN-huayan-x_low.onnx")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_voice(onnx_path: str, label: str):
    json_path = onnx_path + ".json"
    if not os.path.isfile(onnx_path):
        print(f"[WARN] {label} voice not found: {onnx_path}")
        return None
    try:
        voice = PiperVoice.load(
            onnx_path,
            config_path=json_path if os.path.isfile(json_path) else None,
            use_cuda=False,
        )
        print(f"[OK] {label} voice loaded.")
        return voice
    except Exception as e:
        print(f"[ERROR] Failed to load {label} voice: {e}")
        return None


def is_chinese(text: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' for c in text)


def speak(voice, text: str):
    if voice is None:
        print("[ERROR] Voice not loaded, cannot speak.")
        return
    try:
        rate = getattr(getattr(voice, "config", None), "sample_rate", 22050)

        # phonemize() returns list of sentences, each a list of phonemes
        # flatten all sentences into one list for phonemes_to_ids
        phonemes_nested = voice.phonemize(text)
        flat_phonemes = []
        for sentence in phonemes_nested:
            if isinstance(sentence, list):
                flat_phonemes.extend(sentence)
            else:
                flat_phonemes.append(sentence)

        phoneme_ids = voice.phonemes_to_ids(flat_phonemes)
        audio = voice.phoneme_ids_to_audio(phoneme_ids)

        if audio is None or (hasattr(audio, '__len__') and len(audio) == 0):
            print("[ERROR] No audio generated.")
            return

        # audio may be int16 or float32
        if isinstance(audio, np.ndarray):
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio.astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32)
        else:
            audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        sd.play(audio, samplerate=rate)
        sd.wait()

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 50)
    print("  Piper TTS Interactive Test")
    print("  Type text to hear it spoken.")
    print("  Chinese characters -> ZH voice")
    print("  Everything else    -> EN voice")
    print("  Type 'quit' to exit.")
    print("=" * 50)

    print("\nLoading voices ...")
    voice_en = load_voice(VOICE_EN, "EN")
    voice_zh = load_voice(VOICE_ZH, "ZH")

    if not voice_en and not voice_zh:
        print("[ERROR] No voices loaded. Check your voices/ directory.")
        sys.exit(1)

    print("\nReady.\n")

    while True:
        try:
            text = input("Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not text:
            continue
        if text.lower() == "quit":
            print("Exiting.")
            break

        if is_chinese(text):
            print(f"  -> ZH voice")
            speak(voice_zh, text)
        else:
            print(f"  -> EN voice")
            speak(voice_en, text)


if __name__ == "__main__":
    main()