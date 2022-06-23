import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os, sys
import time
import tensorflow as tf
from tcn import TCN

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
    inputs = tf.keras.Input(shape=(None, 12), name="inputs")
    tcn = TCN(return_sequences=True, name="tcn")(inputs)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="outputs")(tcn)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tcn_model")
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["binary_accuracy"])
    model.summary()

    # Fit model
    history = model.fit(x=x_train, y=y_train, batch_size=16, epochs=3, validation_data=(x_test, y_test), shuffle=True)
        
    return
    
if __name__ == "__main__":
    main()
