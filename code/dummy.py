import numpy as np
import os, time
from datetime import datetime
from utils.data_utils import load_data, SAMPLING_FREQUENCY
from custom.data_generator import MyDataGenerator

ROOT_DIR = "/gxfs_work1/cau/sukne964/Mobilise-D"
WIN_LEN = int(10 * SAMPLING_FREQUENCY)

def main():
    start_time = time.time()
    print(f"Start at: {datetime.now()}")
    train_data, val_data, test_data = load_data(path=ROOT_DIR, win_len=WIN_LEN)
    end_time = time.time()
    print(f"Total time: {end_time-start_time:.1f} seconds")

    # Split data and features
    X_train, y1_train, y2_train = train_data
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y1_train: {y1_train.shape}")
    print(f"Shape of y2_train: {y2_train.shape}")
    return

if __name__ == "__main__":
    main()
