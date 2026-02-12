
from time import sleep
from PIL import Image
import sys
import os
import argparse
import pygame  # Import pygame
import subprocess

sys.path.append(os.path.abspath("../Driver"))
from WhisPlay import WhisPlayBoard
board = WhisPlayBoard()
board.set_backlight(50)

global_image_data = None
image_filepath = None

# Initialize pygame mixer
pygame.mixer.init()
sound = None  # Global sound variable
playing = False  # Global variable to track if sound is playing


def press():
    print("pressed")

def held():
    print("held")

def release():
    print("release")

# Register button event
board.on_button_press(press)
board.on_button_hold(held)
board.on_button_release(release)
