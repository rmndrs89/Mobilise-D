import numpy as np
import tensorflow as tf
import os, sys
from utils.data_utils import SAMPLING_FREQUENCY, load_data
from custom.models import get_multi_output_model

# Define root directory
# ROOT_DIR = "/mnt/neurogeriatrics_data/MobiliseD_TVS/rawdata" if sys.platform == "linux" else "Z:\\MobiliseD_TVS\\rawdata"
ROOT_DIR = "/gxfs_work1/cau/sukne964/Mobilise-D"

# Define window length
WIN_LEN = int(6 * SAMPLING_FREQUENCY)

# Set max number of epochs
MAX_EPOCHS = 50

# Define search space of hyperparameters
HYPERPARAMS = {
    "NB_FILTERS": [16, 32,64, 128],
    "KERNEL_SIZE": [3, 5, 7],
    "MAX_DILATION": [d for d in range(2, 8)]
}

# Define logs filepath
LOGS_FILEPATH = "/gxfs_home/cau/sukne964/Mobilise-D/code/training/tuning/runs"

# Define boundaries for the receptive field
R_min, R_max = int(SAMPLING_FREQUENCY), int(6*SAMPLING_FREQUENCY)

# Define callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = "val_loss",
    factor = 0.1,
    patience = 5,
    verbose = 0,
    min_lr = 1e-5
)

def main():
    # Load data
    train_data, val_data, test_data = load_data(ROOT_DIR, win_len=WIN_LEN)
    
    # Split features and labels
    X_train, y1_train, y2_train = train_data
    X_val, y1_val, y2_val = val_data
    y2_train = tf.keras.utils.to_categorical(y2_train, num_classes=5)
    y2_val = tf.keras.utils.to_categorical(y2_val, num_classes=5)
    
    # Print dimensions
    print(f"Shape of X train: {X_train.shape}")
    print(f"Shape of y1 train: {y1_train.shape}")
    print(f"Shape of y2 train: {y2_train.shape}")
    
    # Iterate over the search space
    nb_models = 0  # number of models that have been evaluated
    for kernel_size in HYPERPARAMS["KERNEL_SIZE"]:
        for max_dilation in HYPERPARAMS["MAX_DILATION"]:
            for nb_filters in HYPERPARAMS["NB_FILTERS"]:
                
                # Compute the receptive field
                R = 1 + 2 * (kernel_size - 1) * 1 * sum([2**d for d in range(max_dilation)])
                
                if (R > R_min) and (R < R_max):
                    
                    # Instantiate a model
                    model = get_multi_output_model(
                        nb_channels=X_train.shape[-1],
                        nb_classes=y2_train.shape[-1],
                        **{'nb_filters': nb_filters,
                           'kernel_size': kernel_size,
                           'dilations': [2**d for d  in range(max_dilation)]}
                    )
                    
                    # Train the model for some epochs
                    model.fit(
                        x = X_train,
                        y = {'gait_sequences': y1_train,
                             'gait_events': y2_train},
                        batch_size = 16,
                        epochs = MAX_EPOCHS,
                        validation_data = (X_val, {'gait_sequences': y1_val,
                                                   'gait_events': y2_val}),
                        shuffle = True,
                        verbose = 0,
                        callbacks = [reduce_lr, 
                            tf.keras.callbacks.CSVLogger(
                                os.path.join(LOGS_FILEPATH, f"{nb_models:02d}.csv"), separator=",", append=False
                            )
                        ]
                    )

                    nb_models += 1
                    if nb_models > 1:
                        return                   
    return

if __name__ == "__main__":
    main()
