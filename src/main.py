# #detect button press vs a button hold
# #do stuff with the screen
# #set language up based on button held
# #regulate translations
# #conversation true or false variable

# import time
# from time import sleep
# from unittest import result
# from PIL import Image
# import sys
# import os
# import argparse
# import subprocess
# import argostranslate.package
# import argostranslate.translate

# import whisper
# import pyttsx3


# engine = pyttsx3.init()

# model = whisper.load_model("small")

# #Global Variables
# conversation_active = False
# from_language = "es"  #this needs to be changed relative to their languge of choice/spoken
# to_language = "en" 
# button_held = False


# # Import driver
# sys.path.append(os.path.abspath("../Driver"))
# try:
#     from WhisPlay import WhisPlayBoard
# except ImportError:
#     print("Error: WhisPlay driver not found.")
#     sys.exit(1)

# # Initialize hardware
# board = WhisPlayBoard()
# board.set_backlight(50)

# # Global variables
# img1_data = None  # Recording stage (test1.jpg)
# img2_data = None  # Playback stage (test2.jpg)
# REC_FILE = "data/recorded_voice.wav"
# TRANSLATED_FILE = ""
# recording_process = None


# #upload an image onto the display screen with the correct dimensions
# def load_jpg_as_rgb565(filepath, screen_width, screen_height):
#     """Convert image to RGB565 format supported by the screen"""
#     if not os.path.exists(filepath):
#         print(f"Warning: File not found: {filepath}")
#         return None

#     img = Image.open(filepath).convert('RGB')
#     original_width, original_height = img.size
#     aspect_ratio = original_width / original_height
#     screen_aspect_ratio = screen_width / screen_height

#     if aspect_ratio > screen_aspect_ratio:
#         new_height = screen_height
#         new_width = int(new_height * aspect_ratio)
#         resized_img = img.resize((new_width, new_height))
#         offset_x = (new_width - screen_width) // 2
#         cropped_img = resized_img.crop(
#             (offset_x, 0, offset_x + screen_width, screen_height))
#     else:
#         new_width = screen_width
#         new_height = int(new_width / aspect_ratio)
#         resized_img = img.resize((new_width, new_height))
#         offset_y = (new_height - screen_height) // 2
#         cropped_img = resized_img.crop(
#             (0, offset_y, screen_width, offset_y + screen_height))

#     pixel_data = []
#     for y in range(screen_height):
#         for x in range(screen_width):
#             r, g, b = cropped_img.getpixel((x, y))
#             rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
#             pixel_data.extend([(rgb565 >> 8) & 0xFF, rgb565 & 0xFF])
#     return pixel_data

# #set audio volume
# def set_wm8960_volume_stable(volume_level: str):
#     """Set wm8960 sound card volume"""
#     CARD_NAME = 'wm8960soundcard'
#     DEVICE_ARG = f'hw:{CARD_NAME}'
#     try:
#         subprocess.run(['amixer', '-D', DEVICE_ARG, 'sset', 'Speaker',
#                        volume_level], check=False, capture_output=True)
#         subprocess.run(['amixer', '-D', DEVICE_ARG, 'sset',
#                        'Capture', '100'], check=False, capture_output=True)
#     except Exception as e:
#         print(f"ERROR: Failed to set volume: {e}")


# def start_end_recording():

#     global conversation_active
#     conversation_active = not conversation_active

#     if conversation_active:
#         on_end()
    
#     else:
#         """Enter recording stage: display test1.jpg and start arecord"""
#         global recording_process, img1_data
#         print(">>> Status: Entering recording stage (displaying test1)...")
#         print(">>> Press the button to stop recording and playback...")

#         if img1_data:
#             board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img1_data)

#         # Start recording asynchronously
#         command = ['arecord', '-D', 'hw:wm8960soundcard',
#                 '-f', 'S16_LE', '-r', '16000', '-c', '2', REC_FILE]
#         recording_process = subprocess.Popen(command)

# def start_recording_language_set():
#     """Enter recording stage: display test1.jpg and start arecord"""
#     global recording_process, img1_data

#     print(">>> Status: Entering recording stage (displaying test1)...")
#     print(">>> Press the button to stop recording and playback...")

#     if img1_data:
#         board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img1_data)

#     # Start recording asynchronously
#     command = ['arecord', '-D', 'hw:wm8960soundcard',
#                '-f', 'S16_LE', '-r', '16000', '-c', '2', REC_FILE]
#     button_held = True
#     recording_process = subprocess.Popen(command)

# def release_and_set():
#     global recording_process, from_language, to_language, button_held
#     if button_held:
#         if recording_process and recording_process.poll() is None:
#             recording_process.terminate()
#             recording_process.wait()
#         from_language, to_language = set_languages(REC_FILE)

# def on_end():
#     """Button callback: stop recording -> color change -> display something -> play translated recording"""
#     global recording_process, img1_data, img2_data
#     print(">>> Button pressed!")

#     # 1. Stop recording
#     if recording_process and recording_process.poll() is None:
#         recording_process.terminate()
#         recording_process.wait()

#     # 2. Visual feedback: LED color sequence
#     color_sequence = [(255, 0, 0, 0xF800),
#                       (0, 255, 0, 0x07E0), (0, 0, 255, 0x001F)]
#     for r, g, b, hex_code in color_sequence:
#         board.fill_screen(hex_code)
#         board.set_rgb(r, g, b)
#         sleep(0.3)
#     board.set_rgb(0, 0, 0)

#     # 3. Playback feedback: display test2.jpg and play recorded audio
#     if img2_data:
#         board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img2_data)

#     print(">>> Playing back recording (displaying test2)...")
#     translation = get_translations(from_language, to_language)
#     swap()
#     result = model.transcribe(REC_FILE)
#     print(result["text"])
#     translation_audio = translation.translate_file(result["text"])
#     engine.say(translation_audio)
#     engine.runAndWait()


# def set_languages(audio):
#     global from_language, to_language
#     result = model.transcribe(audio)
#     if from_language != "en": from_language = result["language"]
#     else: to_language = result["language"]
#     translateF, translateC = setup_translation()
#     return translateF, translateC

# def get_translations(fromL, toL):
#     return fromL.get_translation(toL)


# def setup_translation():

#     global from_language, to_language, from_lang, to_lang
#     # Download and install Argos Translate package
#     argostranslate.package.update_package_index()
#     available_packages = argostranslate.package.get_available_packages()
#     available_package = list(
#         filter(
#             lambda x: x.from_code == from_language and x.to_code == to_language, available_packages
#         )
#     )[0]
#     download_path = available_package.download()
#     argostranslate.package.install_from_path(download_path)

#     # Translate
#     installed_languages = argostranslate.translate.get_installed_languages()
#     from_lang = list(filter(
#             lambda x: x.code == from_language,
#             installed_languages))[0]
#     to_lang = list(filter(
#             lambda x: x.code == to_language,
#             installed_languages))[0]
#     translationForeign = from_lang.get_translation(to_lang)
#     translationCommon = to_lang.get_translation(from_lang)
#     return translationForeign, translationCommon

# def swap():
#     global from_language, to_language, from_lang, to_lang
#     from_language, to_language = to_language, from_language
#     return from_lang.get_translation(to_lang), to_lang.get_translation(from_lang)


# # Register callback
# board.on_button_press(start_end_recording)
# board.on_button_release(release_and_set)
# board.on_button_hold(start_recording_language_set)




# # --- Main program ---
# parser = argparse.ArgumentParser()
# parser.add_argument("--img1", default="data/recording.jpg", help="Image for recording stage")
# parser.add_argument("--img2", default="data/playing.jpg", help="Image for playback stage")
# parser.add_argument("--test_wav", default="data/test.wav")
# args = parser.parse_args()

# try:
#     # 1. Load all image data first
#     print("Initializing images...")
#     img1_data = load_jpg_as_rgb565(
#         args.img1, board.LCD_WIDTH, board.LCD_HEIGHT)
#     img2_data = load_jpg_as_rgb565(
#         args.img2, board.LCD_WIDTH, board.LCD_HEIGHT)

#     # 2. Set volume
#     set_wm8960_volume_stable("121")

#     # 3. Play startup audio at launch (displaying test2.jpg)
#     if os.path.exists(args.test_wav):
#         if img2_data:
#             board.draw_image(0, 0, board.LCD_WIDTH,
#                              board.LCD_HEIGHT, img2_data)
#         print(f">>> Playing startup audio: {args.test_wav} (displaying test2)")
#         subprocess.run(
#             ['aplay', '-D', 'plughw:wm8960soundcard', args.test_wav])

#     # 4. After audio finishes, enter recording loop
#     # start_recording()

#     while True:
#         sleep(0.1)

# except KeyboardInterrupt:
#     print("\nProgram exited")
# finally:
#     if recording_process:
#         recording_process.terminate()
#     board.cleanup()
# print("buh-bye!")

#########################################################################################################################################
# GPT VERSION???

# import sys
# import os
# import argparse
# import subprocess
# from time import sleep
# from enum import Enum

# from PIL import Image
# import whisper
# import pyttsx3
# import argostranslate.package
# import argostranslate.translate

# # ======================================================
# # STATE MACHINE
# # ======================================================

# class State(Enum):
#     IDLE = 0
#     RECORDING = 1
#     LANG_SELECT = 2

# state = State.IDLE
# recording_process = None

# # ======================================================
# # CONFIG
# # ======================================================

# REC_FILE = "data/recorded_voice.wav"

# from_lang_code = "es"
# to_lang_code = "en"

# # ======================================================
# # INIT
# # ======================================================

# engine = pyttsx3.init()
# model = whisper.load_model("small")

# # ======================================================
# # HARDWARE DRIVER
# # ======================================================

# sys.path.append(os.path.abspath("../Driver"))
# from WhisPlay import WhisPlayBoard

# board = WhisPlayBoard()
# board.set_backlight(50)

# img_record = None
# img_play = None

# # ======================================================
# # IMAGE UTILS
# # ======================================================

# def load_jpg_as_rgb565(path, w, h):
#     img = Image.open(path).convert("RGB").resize((w, h))
#     buf = []
#     for y in range(h):
#         for x in range(w):
#             r, g, b = img.getpixel((x, y))
#             rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
#             buf.extend([(rgb565 >> 8) & 0xFF, rgb565 & 0xFF])
#     return buf

# # ======================================================
# # AUDIO
# # ======================================================

# def set_volume(level="121"):
#     subprocess.run(
#         ["amixer", "-D", "hw:wm8960soundcard", "sset", "Speaker", level],
#         capture_output=True
#     )

# def start_recording():
#     global recording_process
#     recording_process = subprocess.Popen([
#         "arecord",
#         "-D", "hw:wm8960soundcard",
#         "-f", "S16_LE",
#         "-r", "16000",
#         "-c", "1",
#         REC_FILE
#     ])

# def stop_recording():
#     global recording_process
#     if recording_process and recording_process.poll() is None:
#         recording_process.terminate()
#         recording_process.wait()
#     recording_process = None

# # ======================================================
# # ARGOS TRANSLATION
# # ======================================================

# def setup_translation():
#     argostranslate.package.update_package_index()
#     packages = argostranslate.package.get_available_packages()

#     pkg = next(
#         p for p in packages
#         if p.from_code == from_lang_code and p.to_code == to_lang_code
#     )

#     argostranslate.package.install_from_path(pkg.download())

#     langs = argostranslate.translate.get_installed_languages()
#     src = next(l for l in langs if l.code == from_lang_code)
#     dst = next(l for l in langs if l.code == to_lang_code)

#     return src.get_translation(dst)

# translation = setup_translation()

# # ======================================================
# # TRANSLATE + SPEAK
# # ======================================================

# def translate_and_speak():
#     board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img_play)

#     result = model.transcribe(REC_FILE)
#     text = result["text"]

#     print(">>> Transcribed:", text)

#     translated = translation.translate(text)

#     print(">>> Translated:", translated)

#     engine.say(translated)
#     engine.runAndWait()

# # ======================================================
# # LANGUAGE SET (HOLD)
# # ======================================================

# def finish_language_select():
#     global from_lang_code, translation

#     stop_recording()

#     result = model.transcribe(REC_FILE)
#     detected = result["language"]

#     from_lang_code = detected
#     print(f">>> Source language set to: {from_lang_code}")

#     translation = setup_translation()

# # ======================================================
# # BUTTON CALLBACKS
# # ======================================================

# def on_button_press():
#     global state

#     if state == State.IDLE:
#         state = State.RECORDING
#         board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img_record)
#         start_recording()
#         print(">>> Recording started")

#     elif state == State.RECORDING:
#         stop_recording()
#         state = State.IDLE
#         print(">>> Recording stopped")
#         translate_and_speak()

# def on_button_hold():
#     global state

#     if state == State.IDLE:
#         state = State.LANG_SELECT
#         board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img_record)
#         start_recording()
#         print(">>> Language select recording...")

# def on_button_release():
#     global state

#     if state == State.LANG_SELECT:
#         finish_language_select()
#         state = State.IDLE

# # ======================================================
# # MAIN
# # ======================================================

# parser = argparse.ArgumentParser()
# parser.add_argument("--img_record", default="data/recording.jpg")
# parser.add_argument("--img_play", default="data/playing.jpg")
# args = parser.parse_args()

# img_record = load_jpg_as_rgb565(
#     args.img_record, board.LCD_WIDTH, board.LCD_HEIGHT
# )
# img_play = load_jpg_as_rgb565(
#     args.img_play, board.LCD_WIDTH, board.LCD_HEIGHT
# )

# set_volume()

# board.on_button_press(on_button_press)
# board.on_button_hold(on_button_hold)
# board.on_button_release(on_button_release)

# try:
#     while True:
#         sleep(0.1)
# except KeyboardInterrupt:
#     stop_recording()
#     board.cleanup()
###########################################################################################################################
import sys
import os
import argparse
import subprocess
import time
from time import sleep
from enum import Enum

from PIL import Image
import whisper
import pyttsx3
import argostranslate.package
import argostranslate.translate

# ======================================================
# STATE MACHINE
# ======================================================

class State(Enum):
    IDLE = 0
    RECORDING = 1
    LANG_SELECT = 2
    OUTPUTING = 3

state = State.IDLE
recording_process = None

# Hold timing (seconds)
hold_duration = 1.0
start_time = 0.0
# ======================================================
# CONFIG
# ======================================================

REC_FILE = "data/recorded_voice.wav"

from_lang_code = "es"
to_lang_code = "en"

# ======================================================
# INIT
# ======================================================

engine = pyttsx3.init()
model = whisper.load_model("small")

# ======================================================
# HARDWARE DRIVER
# ======================================================

sys.path.append(os.path.abspath("../Driver"))
from WhisPlay import WhisPlayBoard

board = WhisPlayBoard()
board.set_backlight(50)

img_record = None
img_play = None

# ======================================================
# IMAGE UTILS
# ======================================================

def load_jpg_as_rgb565(filepath, screen_width, screen_height):
    img = Image.open(filepath).convert('RGB')
    original_width, original_height = img.size

    aspect_ratio = original_width / original_height
    screen_aspect_ratio = screen_width / screen_height

    if aspect_ratio > screen_aspect_ratio:
        # Original image is wider, scale based on screen height
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)
        resized_img = img.resize((new_width, new_height))
        # Calculate horizontal offset to center the image
        offset_x = (new_width - screen_width) // 2
        # Crop the image to fit screen width
        cropped_img = resized_img.crop(
            (offset_x, 0, offset_x + screen_width, screen_height))
    else:
        # Original image is taller or has the same aspect ratio, scale based on screen width
        new_width = screen_width
        new_height = int(new_width / aspect_ratio)
        resized_img = img.resize((new_width, new_height))
        # Calculate vertical offset to center the image
        offset_y = (new_height - screen_height) // 2
        # Crop the image to fit screen height
        cropped_img = resized_img.crop(
            (0, offset_y, screen_width, offset_y + screen_height))

    pixel_data = []
    for y in range(screen_height):
        for x in range(screen_width):
            r, g, b = cropped_img.getpixel((x, y))
            rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
            pixel_data.extend([(rgb565 >> 8) & 0xFF, rgb565 & 0xFF])

    return pixel_data


# ======================================================
# AUDIO
# ======================================================

def set_volume(level="121"):
    subprocess.run(
        ["amixer", "-D", "hw:wm8960soundcard", "sset", "Speaker", level],
        capture_output=True
    )

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
# ARGOS TRANSLATION
# ======================================================

def setup_translation():
    argostranslate.package.update_package_index()
    packages = argostranslate.package.get_available_packages()

    pkg = next(
        p for p in packages
        if p.from_code == from_lang_code and p.to_code == to_lang_code
    )

    argostranslate.package.install_from_path(pkg.download())

    langs = argostranslate.translate.get_installed_languages()
    src = next(l for l in langs if l.code == from_lang_code)
    dst = next(l for l in langs if l.code == to_lang_code)

    return src.get_translation(dst)

translation = setup_translation()

# ======================================================
# TRANSLATE + SPEAK
# ======================================================

def translate_and_speak():
    board.draw_image(0, 0, board.LCD_WIDTH, board.LCD_HEIGHT, img_play)

    result = model.transcribe(REC_FILE)
    text = result["text"]

    print(">>> Transcribed:", text)

    translated = translation.translate(text)

    print(">>> Translated:", translated)

    engine.say(translated)
    engine.runAndWait()

# ======================================================
# LANGUAGE MODE
# ======================================================

def finish_language_select():
    global from_lang_code, translation

    stop_recording()

    result = model.transcribe(REC_FILE)
    detected = result["language"]

    from_lang_code = detected
    print(f">>> Source language set to: {from_lang_code}")

    translation = setup_translation()

# ======================================================
# SCREEN OUTPUTS
# ======================================================

def set_image():
    global state
    image_filepath = "imgs/passive.jpg"
    if state == State.IDLE:
        image_filepath = "imgs/passive.jpg"
    elif state == State.RECORDING:
        image_filepath = "imgs/recording.jpg"
    elif state == State.LANG_SELECT:
        image_filepath = "imgs/listening.jpg"
    elif state == State.OUTPUTING:
        image_filepath = "imgs/playing.jpg"

    global_image_data = load_jpg_as_rgb565(
        image_filepath, board.LCD_WIDTH, board.LCD_HEIGHT)
    board.draw_image(0, 0, board.LCD_WIDTH,
                     board.LCD_HEIGHT, global_image_data)
    

# ======================================================
# BUTTON CALLBACKS
# ======================================================

def on_button_press():
    global start_time
    start_time = time.time()


def on_button_release():
    global state, start_time

    duration = time.time() - start_time

    if duration >= hold_duration:

        if state == State.IDLE:
            state = State.LANG_SELECT
            set_image()
            start_recording()
            print("Entered Language Select Mode")

        elif state == State.LANG_SELECT:
            finish_language_select()
            state = State.IDLE
            set_image()
            print("Exited Language Select Mode")

    else:
        if state == State.IDLE:
            state = State.RECORDING
            set_image()
            start_recording()
            print("Recording started")
        elif state == State.RECORDING:
            stop_recording()
            state = State.OUTPUTING
            set_image()
            print("Recording stopped")
            translate_and_speak()
            state = State.IDLE
            set_image()
            print("Returned to Idle Mode")

# ======================================================
# MAIN
# ======================================================

parser = argparse.ArgumentParser()
parser.add_argument("--img_record", default="data/recording.jpg")
parser.add_argument("--img_play", default="data/playing.jpg")
args = parser.parse_args()

img_record = load_jpg_as_rgb565(
    args.img_record, board.LCD_WIDTH, board.LCD_HEIGHT
)
img_play = load_jpg_as_rgb565(
    args.img_play, board.LCD_WIDTH, board.LCD_HEIGHT
)

set_volume()

board.on_button_press(on_button_press)
board.on_button_release(on_button_release)

try:
    while True:
        sleep(0.1)



except KeyboardInterrupt:
    print("Exiting program...")
    stop_recording()
    board.cleanup()