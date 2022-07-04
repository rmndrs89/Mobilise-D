from gc import callbacks
import os
import numpy as np
import tensorflow as tf
from tcn import TCN
from utils.data_utils import SAMPLING_FREQUENCY, split_train_test, get_data_generator
from custom.models import get_model

BASE_DIR = "/gxfs_work1/cau/sukne964/Mobilise-D"
WIN_LEN = int(30*SAMPLING_FREQUENCY)
BATCH_SIZE = 16

# Define model checkpoint
CHECKPOINT_PATH = "/code/my_training"

def main():
    # Split files into training, validation and test set
    train_filenames, val_filenames, test_filenames = split_train_test(path=BASE_DIR)
    print(f"Number of train files: {len(train_filenames)}")
    print(f"Number of valid files: {len(val_filenames)}")
    print(f"Number of test files: {len(test_filenames)}")

    # Define datasets for training and validation
    train_ds = tf.data.Dataset.from_generator(
        get_data_generator(list_files=train_filenames, win_len=WIN_LEN),
        output_signature=(
            tf.TensorSpec(shape=(WIN_LEN, 12), dtype=tf.float32),
            tf.TensorSpec(shape=(WIN_LEN, 1), dtype=tf.float32)
        )
    )

    val_ds = tf.data.Dataset.from_generator(
        get_data_generator(list_files=val_filenames, win_len=WIN_LEN),
        output_signature=(
            tf.TensorSpec(shape=(WIN_LEN, 12), dtype=tf.float32),
            tf.TensorSpec(shape=(WIN_LEN, 1), dtype=tf.float32)
        )
    )

    # Get batched datasets, infiitely repeating for training
    train_ds = train_ds.shuffle(buffer_size=128).batch(batch_size=BATCH_SIZE).repeat()
    val_ds = val_ds.batch(batch_size=BATCH_SIZE)

    # Get model
    model = get_model(num_input_channels=12,
                      **{"nb_filters": 16, 
                         "dilations": (1, 2, 4, 8, 16, 32, 64),
                         "kernel_size": 3})
    model.summary()

    # Model checkpoint
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        CHECKPOINT_PATH,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False
    )
    # Train model on data
    history = model.fit(train_ds, epochs=20, steps_per_epoch=1250, validation_data=val_ds, callbacks=[model_checkpoint])
    return

if __name__ == "__main__":
    main()
