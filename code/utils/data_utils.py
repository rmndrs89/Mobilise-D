import os
import numpy as np
import random
import scipy.io

SAMPLING_FREQUENCY = 64. # Hz

def load_npy(file_name):
    """
    Load the data from the custom .npy file.
    The data columns correspond to the accelerometer and gyroscope
    data of the left and right foot, respectively.
    
    The last two columns represent the labels, 
    with the one-but-last column corresponding to continuous walking periods (CWPs),
    and the last column corresponding to specific gait events (GEs).
    
    For the CWPs the following convention is adopted:
        0 : NULL class, no walking activity
        1 : walking activity
    
    For the GEs the following convention is adopted:
        0 : NULL class, no specific gait event
        1 : left initial contact (ICL),
        2 : left final contact (FCL),
        3 : right initial contact (ICR),
        4 : right final contact (FCR)

    Parameters
    ----------
    file_name : str
        Filename of the .npy data file.

    Returns
    -------
    data, labels : np.ndarray
        Numpy ndarray with the sensor data and labels, respectively.
    """
    
    # Load data from .npy file
    with open(file_name, 'rb') as infile:
        data = np.load(infile)
    
    # Split labels and data
    labels = data[:,-2:]
    data = data[:,:-2]
    return data, labels
        
def load_mat(file_name):
    """
    Load the Mobilise-D data into Python.

    The data is saved as a nested MATLAB struct, and read into a nested Python dictionary.

    The methods are based on:
    https://stackoverflow.com/questions/48970785/complex-matlab-struct-mat-file-read-by-python

    with some adjustments to load the ...
    ... continuous walking periods (ContinuousWalkingPeriod)
    ... micro walking bouts (MicroWB)

    """
    
    def _to_dict(mat_struct):
        """
        Recursively convert nested MATLAB struct objects to Python dictionary entries.
        """
        data_dict = {}
        for field_name in mat_struct._fieldnames:
            elem = mat_struct.__dict__[field_name]
            if isinstance(elem, scipy.io.matlab._mio5_params.mat_struct):
                if not ( field_name in ["ContinuousWalkingPeriod"] ):
                    data_dict[field_name] = _to_dict(elem)
                else:
                    data_dict[field_name] = [_to_dict(elem)]
            elif isinstance(elem, np.ndarray):
                data_dict[field_name] = _to_array(elem, field_name)
            else:
                data_dict[field_name] = elem
        return data_dict
    
    def _to_array(arr, field_name=""):
        """
        TODO: update the description of this method.
        The way the array is converted depends on the data type.
        
        Converts a numpy.ndarray to numpy.ndarray of dtype `float32`.
        """
        if ( arr.dtype in ["<f8", "float64", "float32"] ):
            return arr.astype("float32")
        elif ( arr.dtype in ["uint8", "uint16", "int32"]):
            return arr.astype("float32")
        elif ( arr.dtype in ["O", "object"] ):
            if ( field_name in ["ContinuousWalkingPeriod", "MicroWB"] ):
                return [_to_dict(item) for item in arr]
            elif ( field_name in ["InitialContact_LeftRight", "FinalContact_LeftRight"]):
                return [s for s in arr]
            elif ( field_name in ["Turn_AngularVelocity"] ):
                return [_to_array(item) for item in arr]
            else:
                print(f"No! {field_name} {type(arr.__dict__[field_name])}")
        else:
            print(f"{field_name:s} and {arr.dtype}")
    
    def _check_vars(data_dict):
        """
        Checks if entries in a dictionary are MATLAB struct object.
        If so, then _to_dict() is called to recursively convert them to nested dictionaries.
        """
        for key in data_dict.keys():
            if isinstance(data_dict[key], scipy.io.matlab._mio5_params.mat_struct):
                data_dict[key] = _to_dict(data_dict[key])
            elif isinstance(data_dict[key], scipy.io.matlab._mio5_params.mat_struct):
                data_dict[key] = _to_array(data_dict[key])
        return data_dict
    
    # Call default loadmat from scipy.io module
    data_dict = scipy.io.loadmat(file_name, struct_as_record=False, squeeze_me=True)
    return _check_vars(data_dict)

def create_sequences(values, win_len, step_len):
    """
    Create sequences of equal length for batch processing.

    Parameters
    ----------
    values : (num_time_steps, num_channels) array
        Sensor data organized with N time steps across D channels.
    win_len : int
        Window length given in number of samples.
    step_len : int
        Step length given in number of samples.

    Returns
    -------
    outputs : list
        List of numpy arrays, each of shape (win_len, num_channels).
    """
    outputs = []
    for i in range(0, values.shape[0]-win_len+1, step_len):
        outputs.append(values[i:i+win_len,:])
    return outputs

def load_data(path, sequence_length, overlap=0.0, test_size=None, ):
    """
    Load data from the Mobilise-D technical validation study.

    Parameters
    ----------
    path : str
        Root directory where the data are stored in custom .npy format.
    test_size : float, default=None
        Proportion of the files that are used for the test set.
        If None, it will be set to 0.25.
    """
   
    # Parse input arguments
    seq_len = int(sequence_length*SAMPLING_FREQUENCY)
    step_len = seq_len - int(overlap*seq_len)
    test_size = 0.25 if test_size is None else test_size  # set proportion of test set
    
    # Get list of subject ids
    sub_ids = [sub_id for sub_id in os.listdir(path) if sub_id.startswith('sub-')]
    
    # Split in test and train set
    random.seed(123)
    test_sub_ids = random.sample(sub_ids, int(test_size*len(sub_ids)))
    train_sub_ids = [sub_id for sub_id in sub_ids if sub_id not in test_sub_ids]
    
    # Create train and test data
    train_data = []
    for sub_id in train_sub_ids:
        
        # Load data
        with open(os.path.join(path, sub_id, sub_id+'.npy'), 'rb') as infile:
            data = np.load(infile)
            train_data = train_data + create_sequences(data, win_len=seq_len, step_len=step_len)
    
    test_data = []
    for sub_id in test_sub_ids:
        with open(os.path.join(path, sub_id, sub_id+'.npy'), 'rb') as infile:
            data = np.load(infile)
            test_data = test_data + create_sequences(data, win_len=seq_len, step_len=step_len)
    return np.stack(train_data), np.stack(test_data)