from turtle import back
import numpy as np
import random
import matplotlib.pyplot as plt
import os, sys

from utils.data_utils import SAMPLING_FREQUENCY, load_npy

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

ROOT_DIR = "/mnt/neurogeriatrics_data/MobiliseD_TVS/rawdata" if sys.platform == "linux" else "Z:\\MobiliseD_TVS\\rawdata"
WIN_LEN = int(10 * SAMPLING_FREQUENCY)

def main():

    # Set seed
    random.seed(123)
    
    # Start GUI
    # my_app = MyApp()

    # Get list of subject ids
    sub_ids = [sub_id for sub_id in os.listdir(ROOT_DIR) if sub_id.startswith("sub-")]

    # Divide subject ids in train and test set
    test_sub_ids = random.sample(sub_ids, int(0.3*len(sub_ids)))
    train_sub_ids = [sub_id for sub_id in sub_ids if sub_id not in test_sub_ids]

    # Loop over subject ids in train set
    for (ix_sub_id, sub_id) in enumerate(train_sub_ids):

        # Load data and labels
        data, labels = load_npy(os.path.join(ROOT_DIR, sub_id, sub_id+".npy"))


    data_gen = TimeseriesGenerator(
        data,
        labels,
        length=int(10*SAMPLING_FREQUENCY),
        stride=int(10*SAMPLING_FREQUENCY),
        batch_size=2
    )

    batch_0 = data_gen[0]
    X, y = batch_0
    return
    
if __name__ == "__main__":
    main()