"""
Class for defining constants used
"""

import os

# Audio
SAMPLE_RATE = 44100
N_SAMPLES_IN = 147443
N_SAMPLES_OUT = 16389

# Folders
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")