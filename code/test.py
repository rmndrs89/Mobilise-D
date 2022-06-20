import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os, sys
import time

from utils.data_utils import SAMPLING_FREQUENCY, load_data

ROOT_DIR = "/mnt/neurogeriatrics_data/MobiliseD_TVS/rawdata" if sys.platform == "linux" else "Z:\\MobiliseD_TVS\\rawdata"

def main():

    # Load data
    start_time = time.time()
    train_data, test_data = load_data(path=ROOT_DIR, sequence_length=10., overlap=0., test_size=0.2)
    end_time = time.time()
    print(f'Total time: {end_time - start_time:.2f} sec')
    
    # Split features and labels
    train_features, train_labels = train_data[:,:,:-2], train_data[:,:,-2:]
    train_labels_dict = {
        'gait': np.expand_dims(train_labels[:,:,0], axis=-1),
        'events': np.expand_dims(train_labels[:,:,1], axis=-1)
    }
    
    
    return
    
if __name__ == "__main__":
    main()