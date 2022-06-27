import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os, sys
import time
import tensorflow as tf
from tcn import TCN
import keras_tuner as kt
from custom.hypermodels import TCNHyperModel

from utils.data_utils import SAMPLING_FREQUENCY, load_data

# ROOT_DIR = "/mnt/neurogeriatrics_data/MobiliseD_TVS/rawdata" if sys.platform == "linux" else "Z:\\MobiliseD_TVS\\rawdata"
ROOT_DIR = "/gxfs_work1/cau/sukne964/Mobilise-D"
WIN_LEN = int(10 * SAMPLING_FREQUENCY)  # in samples
EPOCHS = 150        # number of epochs to train for
MAX_TRIALS = 10     # total number of trials to run during the search
EXEC_PER_TRIAL = 5  # number of models that are built and fit for each trials
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss", mode="min")

def main():

    # Load data
    train_data, test_data = load_data(path=ROOT_DIR, win_len=WIN_LEN)

    # Split features and labels
    X_train, y1_train, y2_train = train_data
    X_test, y1_test, y2_test = test_data
    y2_train = tf.keras.utils.to_categorical(y2_train)
    y2_test = tf.keras.utils.to_categorical(y2_test)

    hypermodel = TCNHyperModel(
        nb_channels = X_train.shape[-1],
        nb_classes = y2_train.shape[-1],
        weights_1 = 0.01,
        weights_2 = [0.04, 0.24, 0.24, 0.24, 0.24]
    )

    tuner = kt.RandomSearch(
        hypermodel,
        objective = "val_loss",
        max_trials = MAX_TRIALS,
        executions_per_trial = EXEC_PER_TRIAL,
        overwrite = True,
        directory = "training/runs/00",
        project_name = "bare"
    )
    tuner.search(
        X_train,
        {"gait_sequences": y1_train, "gait_events": y2_train},
        epochs = EPOCHS,
        validation_data = [X_test, {"gait_sequences": y1_test, "gait_events": y2_test}],
        callbacks = [early_stopping_cb],
        verbose = 0
    )

    # Summarize tuner results
    tuner.results_summary()
    return
    
if __name__ == "__main__":
    main()
