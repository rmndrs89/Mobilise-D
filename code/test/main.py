import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import matplotlib.pyplot as plt
# from data_generator import wrapper_fn
from datasets import get_batch_from_file

import tensorflow as tf

ROOT_DIR = "/mnt/neurogeriatrics_data/MobiliseD_TVS/rawdata"
SAMPLING_FREQUENCY = 64.  # Hz
WIN_LEN = int(10 * 60 * SAMPLING_FREQUENCY)

def main():
    
    # Get list of subject ids
    sub_ids = [sub_id for sub_id in os.listdir(ROOT_DIR) if sub_id.startswith("sub-")]
    
    # Parse first two files
    arr_1 = get_batch_from_file(
        filename=os.path.join(ROOT_DIR, sub_ids[0], sub_ids[0]+'.npy'),
        win_len=WIN_LEN
    )
    arr_2 = get_batch_from_file(
        filename=os.path.join(ROOT_DIR, sub_ids[1], sub_ids[1]+'.npy'),
        win_len=WIN_LEN
    )
    
    # Create datasets from the data arrays
    a = tf.data.Dataset.from_tensor_slices(arr_1)
    b = tf.data.Dataset.from_tensor_slices(arr_2)
    return
        

if __name__ == "__main__":
    main()
