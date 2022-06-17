from turtle import back
import numpy as np
import random
import matplotlib.pyplot as plt
import os, sys

from utils.data_utils import SAMPLING_FREQUENCY, load_npy

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

ROOT_DIR = "/mnt/neurogeriatrics_data/MobiliseD_TVS/rawdata" if sys.platform == "linux" else "Z:\\MobiliseD_TVS\\rawdata"
SEQUENCE_LENGTH = int(10 * SAMPLING_FREQUENCY)

def main():

    # Set seed
    random.seed(123)
    
    # Start GUI
    # my_app = MyApp()

    # Get list of subject ids
    sub_ids = [sub_id for sub_id in os.listdir(ROOT_DIR) if sub_id.startswith("sub-")]
    filenames_list = [[os.path.join(ROOT_DIR, sub_id, filename) for filename in os.listdir(os.path.join(ROOT_DIR, sub_id))][0] for sub_id in sub_ids]
    
    # Accumulate data over multiple files
    data = []
    for idx_file in range(len(filenames_list[:5])):
                
        # Load data
        with open(filenames_list[idx_file], 'rb') as infile:
            values = np.load(infile)
        
        for i in range(values.shape[0] - SEQUENCE_LENGTH + 1):
            data.append(values[i:i+SEQUENCE_LENGTH,:])

    return
    
if __name__ == "__main__":
    main()