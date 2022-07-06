import numpy as np
import tensorflow as tf
import os
from utils.data_utils import SAMPLING_FREQUENCY, get_data_generator, split_train_test
from custom.models import get_multi_output_model

# Define root directory
# ROOT_DIR = "/mnt/neurogeriatrics_data/MobiliseD_TVS/rawdata" if sys.platform == "linux" else "Z:\\MobiliseD_TVS\\rawdata"
ROOT_DIR = "/gxfs_work1/cau/sukne964/Mobilise-D"

# Define window length
WIN_LEN = int(10 * SAMPLING_FREQUENCY)

# Define number of input channels
NUM_INPUT_CHANNELS = 12

# Set max number of epochs
MAX_EPOCHS = 50
BATCH_SIZE = 16

# Define model checkpoint pathlogs filepath
CHECKPOINT_FILEPATH = "/gxfs_home/cau/sukne964/Mobilise-D/code/train/tune/04"

# Define callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = 5,
    verbose = 0,
    min_lr = 1e-5
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = 5,
    verbose = 0,
    mode = "min"
)

csv_logger = tf.keras.callbacks.CSVLogger(
    filename = os.path.join(CHECKPOINT_FILEPATH, "training.log"),
    separator = ",",
    append = False
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    CHECKPOINT_FILEPATH,
    monitor = "val_loss",
    verbose = 0,
    save_best_only = True,
    save_weights_only = False
)

def main():
    
    # Get list of filenames for train, val and test set
    train_filenames, val_filenames, test_filenames = split_train_test(path=ROOT_DIR, test_size=0.2)
    
    # Get tf.data.Dataset for train and val set
    train_ds = tf.data.Dataset.from_generator(
        get_data_generator(list_files=train_filenames,
                           win_len=WIN_LEN),
        output_signature=(
            tf.TensorSpec(shape=(WIN_LEN, NUM_INPUT_CHANNELS), dtype=tf.float32),
            {"gait_sequences": tf.TensorSpec(shape=(WIN_LEN, 1), dtype=tf.float32),
             "gait_events": tf.TensorSpec(shape=(WIN_LEN, 5), dtype=tf.float32)}
        )
    )
    val_ds = tf.data.Dataset.from_generator(
        get_data_generator(list_files=val_filenames,
                           win_len=WIN_LEN),
        output_signature=(
            tf.TensorSpec(shape=(WIN_LEN, NUM_INPUT_CHANNELS), dtype=tf.float32),
            {"gait_sequences": tf.TensorSpec(shape=(WIN_LEN, 1), dtype=tf.float32),
             "gait_events": tf.TensorSpec(shape=(WIN_LEN, 5), dtype=tf.float32)}
        )
    )
    
    # Get batched dataset, infinitely repeating for the training set
    train_ds = train_ds.shuffle(buffer_size=128).batch(batch_size=BATCH_SIZE).repeat()
    val_ds = val_ds.batch(batch_size=BATCH_SIZE)
    
    # Get a multi-output model, simulatenous predicting
    # continuous walking periods and discrete gait events
    model = get_multi_output_model(
        num_input_channels=NUM_INPUT_CHANNELS,
        **{"nb_filters": 32,
           "dilations": [1, 2, 4, 8, 16, 32, 64],
           "kernel_size": 5}
    )
       
    # Train model on data
    history = model.fit(
        train_ds,
        epochs = MAX_EPOCHS,
        steps_per_epoch = 1000,
        validation_data = val_ds,
        callbacks = [early_stopping,
                     csv_logger,
                     model_checkpoint]
    )
    return

if __name__ == "__main__":
    main()
