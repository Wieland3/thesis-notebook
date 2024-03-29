"""
This file implements utility functions for dealing with audio.
"""

import numpy as np
from src import constants


def select_preview_snippet(audio, start, end):
    """
    Selects preview snippet for the audio. If start and end is None, original audio returns.
    :param audio: Audio to operate on
    :param start: Start in seconds
    :param end: End in seconds
    :return: Snipped from audio between start:end
    """
    if start and end:
        audio.sig = audio.sig[start * audio.sr:end * audio.sr]
    return audio


def stereo_to_mono(audio_stereo):
    """
    Converts a stereo audio signal to mono by averaging the channels.
    :param audio_stereo: A numpy array with shape (samples, 2) representing the stereo audio.
    :return: A numpy array with shape (samples,) representing the mono audio.
    """
    if audio_stereo.ndim != 2 or audio_stereo.shape[1] != 2:
        raise ValueError("Input audio must be a stereo signal")

    audio_mono = np.mean(audio_stereo, axis=1, keepdims=True)

    return audio_mono


def zero_pad(array):
    """
    Pads the input audio array with zeros both at the beginning and at the end in the time domain.
    Works for both mono and stereo signals.
    :param array: Input audio array, which can be mono (1D) or stereo (2D).
    :return: Padded audio array.
    """
    pad_width = (constants.N_SAMPLES_IN, constants.N_SAMPLES_IN)

    if array.ndim == 1:
        padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)
    elif array.ndim == 2:
        padded_array = np.pad(array, (pad_width, (0, 0)), mode='constant', constant_values=0)
    else:
        raise ValueError("Input array must be either mono or stereo.")

    return padded_array


def center_crop(array, num_samples=constants.N_SAMPLES_OUT):
    """
    Center crops an audio array.
    :param array: Array to operate on
    :param num_samples: length of the cropped audio
    :return: numpy array with cropped audio
    """
    start = (array.shape[0] - num_samples) // 2
    end = start + num_samples
    return array[start:end]
