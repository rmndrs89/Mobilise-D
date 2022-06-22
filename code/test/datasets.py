import os
import numpy as np

def get_batch_from_file(filename, win_len, step_len=None):
    """
    Get a batch of data examples for the given data file.

    Parameters
    ----------
    filename : str
        Absolute or relative path to the data file (.npy).
    win_len : int
        Length of the window, in number of time steps.
    step_len : int, optional
        Length of the step for sliding the window, by default None

    Returns
    -------
    _ : numpy array, (nb_examples, nb_time_steps, nb_channels)
        A batch of sensor data where each example exists of
        number of time steps equal to the given window length.
    """
    assert isinstance(win_len, int)
    
    # Set step length
    step_len = win_len if step_len is None else int(step_len)
    
    # Get data
    outputs = []
    with open(filename, 'rb') as infile:
        data = np.load(infile)  # features and labels
        
    # Create sequences of equal length
    for idx in range(0, data.shape[0]-win_len+1, step_len):
        outputs.append(data[idx:idx+win_len,:-2])
    return np.stack(outputs)  # (nb_examples, nb_time_steps, nb_channels)