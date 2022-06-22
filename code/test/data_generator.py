import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

def _get_x_y(filepath, win_len, step_len=None, batch_size=1):
    assert isinstance(win_len, int)
    
    # Set step length
    step_len = win_len if step_len is None else int(step_len)
    
    # Get list of subject ids
    sub_ids = [sub_id for sub_id in os.listdir(filepath) if sub_id.startswith("sub-")]
    
    # Iterating over the files (subjects ids)
    while True:
        for idx_sub_id, sub_id in enumerate(sub_ids):
            
            # Load file
            with open(os.path.join(filepath, sub_id, sub_id+".npy"), 'rb') as infile:
                data = np.load(infile)  # features and labels
                
                # Iterate over current time series
                for idx in range(0, data.shape[0] - win_len + 1, step_len):
                    yield os.path.join(filepath, sub_id, sub_id+".npy"), data[idx:idx+win_len,:-2], data[idx:idx+win_len,-2][..., np.newaxis]

def wrapper_fn(filepath, win_len, step_len=None):
    def get_x_y():
        assert isinstance(win_len, int)
        
        # Set step length
        step_len = win_len if step_len is None else int(step_len)
        
        # Get list of subject ids
        sub_ids = [sub_id for sub_id in os.listdir(filepath) if sub_id.startswith("sub-")]
        
        # Iterating over the files (subjects ids)
        while True:
            for idx_sub_id, sub_id in enumerate(sub_ids):
                
                # Load file
                with open(os.path.join(filepath, sub_id, sub_id+".npy"), 'rb') as infile:
                    data = np.load(infile)  # features and labels
                    
                    # Iterate over current time series
                    for idx in range(0, data.shape[0] - win_len + 1, step_len):
                        yield data[idx:idx+win_len,:-2]
    return get_x_y