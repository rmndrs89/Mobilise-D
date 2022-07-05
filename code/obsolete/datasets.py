import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tcn import TCN

BASE_PATH = "/mnt/neurogeriatrics_data/MobiliseD_TVS/deriveddata"

def get_model(input_dims):
    inputs = tf.keras.layers.Input(shape=(None, input_dims), name="inputs")
    hidden = TCN(nb_filters=16, dilations=(1, 2, 4, 8), padding="same", return_sequences=True, name="tcn")(inputs)
    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid", name="outputs")(hidden)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="tcn_model")
    model.compile(loss="mse", optimizer="adam")
    return model

def main():
    
    # Get list of train files
    train_path = os.path.join(BASE_PATH, "train")
    train_files_ds = tf.data.Dataset.list_files(os.path.join(train_path, "*.npy"))
    
    for elem in train_files_ds.take(3):
        print(elem)
    return

if __name__ == "__main__":
    main()
