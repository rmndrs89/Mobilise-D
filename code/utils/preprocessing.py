import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import filtfilt

def _resample(X, fs_old, fs_new):
    """
    Resample sensor data from original sampling frequency to desired sampling frequency.

    Parameters
    ----------
    X : (N, D) numpy array
        Sensor data with N time steps across D channels.
    fs_old : int, float
        Original sampling frequency (in Hz).
    fs_new : int, float
        New sampling frequency (in Hz).
    """
    
    # Determine original time points
    ts = np.arange(X.shape[0])/fs_old
    
    # Account for missing values
    ts = ts[np.logical_not(np.any(np.isnan(X), axis=1))]
    X = X[np.logical_not(np.any(np.isnan(X), axis=1)),:]
    
    # Linear interpolate
    f = interp1d(ts, X, kind="linear", axis=0, fill_value="extrapolate")
    
    # Determine query time points
    tq = np.arange(X.shape[0]/fs_old*fs_new)/fs_new
    return f(tq)