import numpy as np
from src import predict
import soundfile as sf
from src.time_noise_gate import TimeNoiseGate


def run_and_save_prediction(model_name, vocals, clean_sources, use_noise_gate, threshold, song_name):
    """
    Function for predicting a song
    :param model_name: Name of model to use
    :param vocals: Pya asig for vocal
    :param clean_sources: Pya asig for clean sources
    :param use_noise_gate: Boolean if noise gate as post processing should be used
    :param threshold: Threshold for that noise gate
    :param song_name: Name to save predicted song under
    :return: None
    """

    exploited = False
    X = vocals.sig
    length = X.shape[0]
    if model_name == "exploited":
        exploited = True
    if exploited:
        X = np.hstack([vocals.sig.reshape(-1,1), clean_sources.sig.reshape(-1,1)])

    prediction = predict.predict_song(X, exploited, model_name)

    if use_noise_gate:
        noise_gate = TimeNoiseGate(threshold)
        prediction = noise_gate.process(prediction)

    prediction = prediction.squeeze(axis=-1)
    length_difference = prediction.shape[0] - length
    remove_samples = length_difference // 2
    prediction = prediction[remove_samples:-remove_samples]
    sf.write("./predictions/" + song_name + ".wav", prediction, vocals.sr)
    print("Saved Prediction in Predictions Folder")
