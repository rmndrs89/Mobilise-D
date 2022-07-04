import os
import numpy as np
import tensorflow as tf
from tcn import TCN
from custom.losses import MyWeightedBinaryCrossentropy

BASE_DIR = "/gxfs_work1/cau/sukne964/Mobilise-D"
SAMPLING_FREQUENCY = 64.  # in Hz
WIN_LEN = int(30*SAMPLING_FREQUENCY)

def split_train_test(path, test_size=0.25, seed=None):
    # Map subject ids to specific study site
    i2s = {"10": "UNEW", "20": "USFD", "30": "TASMC", "40": "CAU", "50":"RBMF"}

    # Set seed
    seed = np.random.seed(123) if seed is None else np.random.seed(seed)

    # Get a list of subject ids
    sub_ids = [sub_id for sub_id in os.listdir(path) if sub_id.startswith("sub-")]
    sub_ids_per_site = {v: [] for _, v in i2s.items()}
    for sub_id in sub_ids:
        sub_ids_per_site[i2s[sub_id[4:6]]].append(sub_id)

    # Split subject ids in training, validation and test set
    train_sub_ids, val_sub_ids, test_sub_ids = [], [], []
    for _, ids in sub_ids_per_site.items():
        num = int(len(ids)*test_size)
        indices = np.arange(len(ids))
        np.random.shuffle(indices)
        test_sub_ids += [ids[idx] for idx in indices[:num]]
        val_sub_ids += [ids[idx] for idx in indices[num:2*num]]
        train_sub_ids += [ids[idx] for idx in indices[2*num:]]

    # Return list of filenames for each set
    return [os.path.join(path, sub_id, sub_id+".npy") for sub_id in train_sub_ids], \
        [os.path.join(path, sub_id, sub_id+".npy") for sub_id in val_sub_ids], \
        [os.path.join(path, sub_id, sub_id+".npy") for sub_id in test_sub_ids]

def get_data_generator(list_files, win_len, step_len=None):
    step_len = win_len if step_len is None else step_len

    def data_gen():
        for filename in list_files:
            with open(filename, 'rb') as infile:
                data = np.load(infile)
            for idx in range(0, data.shape[0] - win_len + 1, step_len):
                yield data[idx:idx+win_len,:-2], data[idx:idx+win_len,-2][..., np.newaxis]
    return data_gen

def get_model(num_input_channels):
    inputs = tf.keras.layers.Input(shape=(None, num_input_channels), name="inputs")
    tcn = TCN(
        nb_filters=16, 
        kernel_size=5,
        dilations=(1, 2, 4, 8, 16, 32, 64),
        use_skip_connections=True,
        use_batch_norm=True,
        return_sequences=True,
        padding="same",
        name="tcn")(inputs)
    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid", name="outputs")(tcn)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="tcn_model")
    model.compile(
        loss=MyWeightedBinaryCrossentropy(weight=0.01),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    return model

def main():
    train_filenames, val_filenames, test_filenames = split_train_test(path=BASE_DIR)
    print(f"Number of train files: {len(train_filenames)}")
    print(f"Number of valid files: {len(val_filenames)}")
    print(f"Number of test files: {len(test_filenames)}")

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

    BATCH_SIZE = 16
    train_ds = train_ds.batch(batch_size=BATCH_SIZE).repeat()
    val_ds = val_ds.batch(batch_size=BATCH_SIZE)

    model = get_model(num_input_channels=12)
    model.summary()

    history = model.fit(train_ds, epochs=20, steps_per_epoch=1250, validation_data=val_ds)
    return

if __name__ == "__main__":
    main()
