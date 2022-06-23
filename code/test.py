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
WIN_LEN = int(10 * 60 * SAMPLING_FREQUENCY)

def main():

    # Load data
    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = load_data(path=ROOT_DIR, win_len=WIN_LEN)
    end_time = time.time()

    # Create model
    # inputs = tf.keras.Input(shape=(None, 12), name="inputs")
    # tcn = TCN(return_sequences=True, name="tcn")(inputs)
    # outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="outputs")(tcn)
    # model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tcn_model")
    # model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["binary_accuracy"])
    # model.summary()

    # Fit model
    # history = model.fit(x=x_train, y=y_train, batch_size=16, epochs=3, validation_data=(x_test, y_test), shuffle=True)

    hypermodel = TCNHyperModel(
        nb_channels = x_train.shape[-1]
    )

    tuner = kt.RandomSearch(
        hypermodel,
        objective = "val_loss",
        max_trials = 3,
        executions_per_trial = 2,
        overwrite = True,
        directory = "training/runs/00",
        project_name = "bare"
    )
    tuner.search(
        x_train,
        y_train,
        epochs = 5,
        validation_data = [x_test, y_test],
        verbose = 0
    )
    print(f"{tuner.get_best_hyperparameters()[0]}")
        
    return
    
if __name__ == "__main__":
    main()
