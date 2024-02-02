import tensorflow as tf
import numpy as np
from src import constants
from src import audio_utils


def sample_generator(song, vocal):
    """
    Sample generator
    :param song: song to operate one
    :param vocal: vocal to operate on
    :return: yields the chunks from the song
    """
    l, r = 0, constants.N_SAMPLES_IN
    step = constants.N_SAMPLES_OUT

    while r <= song.shape[0]:
        X_chunk = np.array(song[l:r])
        y_chunk = audio_utils.center_crop(vocal[l:r])
        yield X_chunk, y_chunk

        l += step
        r += step


def load_model(model_name):
    """
    Loads correct models.
    :param exploited: If exploited Model should be used
    :return: keras Model
    """
    if model_name == "musdb":
        checkpoint_path = constants.CHECKPOINTS_DIR + "/full_train/cp.ckpt"
    elif model_name == "exploited":
        checkpoint_path = constants.CHECKPOINTS_DIR + "/exploit_full_train/cp.ckpt"
    elif model_name == "unexploited":
        checkpoint_path = constants.CHECKPOINTS_DIR + "/full_train_artificial_unexploited/cp.ckpt"
    else:
        raise ValueError("Invalid Model Name")

    return tf.keras.models.load_model(checkpoint_path)


def predict_song(X, exploited, model_name):
    """
    Predicts an entire song and returns prediction
    :param X: Song to predict (if exploited needs to contain clean sources in right channel)
    :param exploited: If exploited models should be used
    :return: Unbleeded song
    """
    pred = []

    model = load_model(model_name)
    X = audio_utils.zero_pad(X)

    if X.ndim == 1:
        X = np.expand_dims(X, axis=-1)

    for i, (X_chunk, _) in enumerate(sample_generator(X, X)):
        X_chunk_batch = np.expand_dims(X_chunk, axis=0)
        y_pred_chunk = model.predict(X_chunk_batch)['vocals']
        y_pred_chunk = y_pred_chunk.squeeze(0)
        pred.append(y_pred_chunk)

    pred = np.concatenate(pred, axis=0)

    if exploited:
        pred = audio_utils.stereo_to_mono(pred)

    return pred
