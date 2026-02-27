import sys
import os
import subprocess
import time
import json
import wave
from enum import Enum
from time import sleep

from PIL import Image
import pyttsx3
import vosk
import ctranslate2

# ======================================================
# STATE MACHINE
# ======================================================

class State(Enum):
    IDLE = 0
    LISTENING = 1
    PROCESSING = 2

state = State.IDLE

recording_process = None
start_time = 0.0
hold_duration = 1.0
last_activity_time = 0.0

CONVO_TIMEOUT = 15
REC_FILE = "data/recorded_voice.wav"

# True = English → Chinese
# False = Chinese → English
eng_to_cn = True

# ======================================================
# INIT MODELS
# ======================================================

vosk_model_en = vosk.Model("models/vosk-en")
vosk_model_cn = vosk.Model("models/vosk-cn")

translator_en_zh = ctranslate2.Translator("models/opus-en-zh-ct2")
translator_zh_en = ctranslate2.Translator("models/opus-zh-en-ct2")

engine = pyttsx3.init()

# ======================================================
# HARDWARE
# ======================================================

sys.path.append(os.path.abspath("../Driver"))
from WhisPlay import WhisPlayBoard

board = WhisPlayBoard()
board.set_backlight(50)

# ======================================================
# IMAGE UTIL
# ======================================================

def load_jpg_as_rgb565(filepath, w, h):
    img = Image.open(filepath).convert("RGB")
    img = img.resize((w, h))

    data = []
    for y in range(h):
        for x in range(w):
            r, g, b = img.getpixel((x, y))
            rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
            data.extend([(rgb565 >> 8) & 0xFF, rgb565 & 0xFF])
    return data

# Load images
img_idle = load_jpg_as_rgb565("imgs/passive.jpg", board.LCD_WIDTH, board.LCD_HEIGHT)
img_record = load_jpg_as_rgb565("imgs/recording.jpg", board.LCD_WIDTH, board.LCD_HEIGHT)
img_play = load_jpg_as_rgb565("imgs/playing.jpg", board.LCD_WIDTH, board.LCD_HEIGHT)

def update_display():
    if state == State.IDLE:
        board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img_idle)
    elif state == State.LISTENING:
        board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img_record)
    elif state == State.PROCESSING:
        board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img_play)

# ======================================================
# AUDIO
# ======================================================

def start_recording():
    global recording_process
    recording_process = subprocess.Popen([
        "arecord",
        "-D", "hw:wm8960soundcard",
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "1",
        REC_FILE
    ])

def stop_recording():
    global recording_process
    if recording_process and recording_process.poll() is None:
        recording_process.terminate()
        recording_process.wait()
    recording_process = None

# ======================================================
# TRANSCRIPTION
# ======================================================

def transcribe():
    wf = wave.open(REC_FILE, "rb")

    model = vosk_model_en if eng_to_cn else vosk_model_cn
    rec = vosk.KaldiRecognizer(model, wf.getframerate())

    text_result = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            text_result += res.get("text", "") + " "

    final_res = json.loads(rec.FinalResult())
    text_result += final_res.get("text", "")

    return text_result.strip()

# ======================================================
# TRANSLATION
# ======================================================

def translate_text(text):
    tokens = text.split()

    if eng_to_cn:
        result = translator_en_zh.translate_batch([tokens])
    else:
        result = translator_zh_en.translate_batch([tokens])

    return " ".join(result[0].hypotheses[0])

# ======================================================
# PROCESS ONE TURN
# ======================================================

def process_turn():
    global eng_to_cn, state, last_activity_time

    state = State.PROCESSING
    update_display()

    text = transcribe()
    print("Input:", text)

    if not text:
        state = State.LISTENING
        update_display()
        start_recording()
        return

    translated = translate_text(text)
    print("Output:", translated)

    engine.say(translated)
    engine.runAndWait()

    # Swap direction automatically
    eng_to_cn = not eng_to_cn

    last_activity_time = time.time()

    state = State.LISTENING
    update_display()
    start_recording()

# ======================================================
# BUTTON HANDLING
# ======================================================

def on_button_press():
    global start_time
    start_time = time.time()

def on_button_release():
    global state, eng_to_cn, last_activity_time

    duration = time.time() - start_time

    # If idle → start conversation
    if state == State.IDLE:

        if duration >= hold_duration:
            eng_to_cn = True
            print("Conversation started: English → Chinese")
        else:
            eng_to_cn = False
            print("Conversation started: Chinese → English")

        state = State.LISTENING
        last_activity_time = time.time()
        update_display()
        start_recording()
        return

    # If listening → stop, translate, restart
    if state == State.LISTENING:
        stop_recording()
        process_turn()

# ======================================================
# MAIN LOOP
# ======================================================

board.on_button_press(on_button_press)
board.on_button_release(on_button_release)

update_display()

try:
    while True:

        if state != State.IDLE:
            if time.time() - last_activity_time > CONVO_TIMEOUT:
                print("Conversation timeout. Returning to idle.")
                stop_recording()
                state = State.IDLE
                update_display()

        sleep(0.1)

except KeyboardInterrupt:
    stop_recording()
    board.cleanup()