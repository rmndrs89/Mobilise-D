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
WIN_LEN = int(10 * SAMPLING_FREQUENCY)
EPOCHS = 150         # number of epochs to train for
MAX_TRIALS = 20      # total number of trials to run during the search
EXEC_PER_TRIAL = 10  # number of models that are built and fit for each trials
PATIENCE = 5
CHECKPOINT_PATH = "/gxfs_home/cau/sukne964/Mobilise-D/code/training/checkpoints"
TUNER_PATH = "/gxfs_home/cau/sukne964/Mobilise-D/code/trainig/hptuning"

# Define callbacks
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=PATIENCE, monitor="val_loss", mode="min")
model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=False, save_best_only=True, monitor="val_loss", mode="min")
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_lr=1e-5)

def main():
    
    # Load data
    train_data, val_data, test_data = load_data(path=ROOT_DIR, win_len=WIN_LEN)

    # Split features and labels
    X_train, y1_train, y2_train = train_data
    X_val, y1_val, y2_val = val_data
    X_test, y1_test, y2_test = test_data
    y2_train = tf.keras.utils.to_categorical(y2_train)
    y2_val = tf.keras.utils.to_categorical(y2_val)
    y2_test = tf.keras.utils.to_categorical(y2_test)

    # Instantiate a hypermodel
    hypermodel = TCNHyperModel(
        nb_channels = X_train.shape[-1],
        nb_classes = y2_train.shape[-1],  # number of classes for events detection
        weights_1 = 0.01,
        weights_2 = [0.04, 0.24, 0.24, 0.24, 0.24]
    )

    # Instantiate a tuner
    tuner = kt.RandomSearch(
        hypermodel,
        objective = "val_loss",
        max_trials = MAX_TRIALS,
        executions_per_trial = EXEC_PER_TRIAL,
        overwrite = True,
        directory = TUNER_PATH,
        project_name = "Mobilise-D"
    )

    # Use tuner to explore the search space
    tuner.search(
        X_train,
        {"gait_sequences": y1_train, "gait_events": y2_train},
        batch_size = 16,
        epochs = 30,
        validation_data = [X_val, {"gait_sequences": y1_val, "gait_events": y2_val}],
        callbacks = [reduce_lr_cb],
        verbose = 0
    )

    # Query tuner results
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Found the optimum hyperparameters:")
    print(f"Number of conv filters: {2**best_hps.get('nb_filters')}")
    print(f"Dilations: {[2**d for d in range(best_hps.get('dilations'))]}")
    print(f"Dropout rate: {best_hps.get('dropout_rate')}")

    # Instantiate a model based on the best hyperparams
    best_model = tuner.hypermodel.build(best_hps)
    best_model.summary()

    # Train best model on training data
    history = best_model.fit(
        x = X_train,
        y = {'gait_sequences': y1_train, 'gait_events': y2_train},
        batch_size = 16,
        epochs = EPOCHS,
        validation_data = (X_val, {'gait_sequences': y1_val, 'gait_events': y2_val}),
        callbacks = [early_stopping_cb, model_checkpoint_cb],
        verbose = 0
    )

    return
    
if __name__ == "__main__":
    main()
