import os
import numpy as np
import tensorflow as tf
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

def create_sequences(data, win_len, step_len):
    """
    Create sequences of equal length for batch processing.

    Parameters
    ----------
    data : (nb_time_steps, nb_channels) array
        Sensor data organized with N time steps across D channels.
    win_len : int
        Window length given in number of samples.
    step_len : int
        Step length given in number of samples.

    Returns
    -------
    sequences : list
        List of numpy arrays, each of shape (win_len, num_channels).
    """
    # Preliminary checks
    assert isinstance(win_len, int)
    step_len = win_len if step_len is None else step_len

    # Iterate over data array
    sequences = []
    for idx in range(0, data.shape[0] - win_len + 1, step_len):

        # Append current data segment to list
        sequences.append(data[idx:idx+win_len,:])
    return sequences

def create_batch_sequences(filenames, win_len, step_len=None):
    """
    Create batches of sequences of equal length for batch processing.

    Parameters
    ----------
    data : (nb_time_steps, nb_channels) array
        Sensor data organized with N time steps across D channels.
    win_len : int
        Window length given in number of samples.
    step_len : int
        Step length given in number of samples.

    Returns
    -------
    batch_sequences : list
        List of numpy arrays, each of shape (win_len, num_channels).
    """
    assert isinstance(win_len, int)

    # Set step length
    step_len = win_len if step_len is None else step_len

    # Iterate over filenames
    batch_sequences = []
    for _, filename in enumerate(filenames):
        
        # Load data from file
        with open(filename, 'rb') as infile:
            data = np.load(infile)

        # Create sequences of equal length
        sequences = create_sequences(data, win_len=win_len, step_len=step_len)

        # Add to overall list of sequences
        batch_sequences += sequences
    return batch_sequences

def load_data(path, win_len, step_len=None, test_size=None):
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
    # Define mapping from subject id to study site
    id_2_site = {'10': 'UNEW', '20': 'USFD', '30': 'TASMC', '40': 'CAU', '50': 'RBMF'}
   
    # Parse input arguments
    step_len = win_len if step_len is None else step_len
    test_size = 0.25 if test_size is None else test_size
        
    # Get list of subject ids
    sub_ids = [sub_id for sub_id in os.listdir(path) if sub_id.startswith('sub-')]
    
    # Split subject ids per study site
    sub_ids_per_site = {v: [] for k, v in id_2_site.items()}
    for sub_id in sub_ids:
        sub_ids_per_site[id_2_site[sub_id[4:6]]].append(sub_id)
    
    # Split in test, validation and train set
    random.seed(123)
    test_sub_ids, val_sub_ids, train_sub_ids = [], [], []
    for _, ids in sub_ids_per_site.items():
        num_ids = int(len(ids)*test_size)  # total number of subjects for current study site
        idxs = np.arange(len(ids))         # arange list of indexes
        np.random.shuffle(idxs)            # randomly shuffle
        test_sub_ids += [ids[idx] for idx in idxs[:num_ids]]  # take proportion of dataset for testing, validation, and training
        val_sub_ids += [ids[idx] for idx in idxs[num_ids:2*num_ids]]
        train_sub_ids += [sub_id for sub_id in ids if (sub_id not in test_sub_ids) and (sub_id not in val_sub_ids)]
    
    # test_sub_ids = random.sample(sub_ids, int(test_size*len(sub_ids)))
    # train_sub_ids = [sub_id for sub_id in sub_ids if sub_id not in test_sub_ids]
    
    # Create train and test data
    train_filenames = [os.path.join(path, sub_id, sub_id+'.npy') for sub_id in train_sub_ids]
    train_dataset = np.stack(create_batch_sequences(train_filenames, win_len=win_len, step_len=step_len))
    val_filenames = [os.path.join(path, sub_id, sub_id+'.npy') for sub_id in val_sub_ids]
    val_dataset = np.stack(create_batch_sequences(val_filenames, win_len=win_len, step_len=step_len))
    test_filenames = [os.path.join(path, sub_id, sub_id+'.npy') for sub_id in test_sub_ids]
    test_dataset = np.stack(create_batch_sequences(test_filenames, win_len=win_len, step_len=step_len))
    return (train_dataset[:,:,:-2], train_dataset[:,:,-2][..., np.newaxis], train_dataset[:,:,-1][..., np.newaxis]), \
        (val_dataset[:,:,:-2], val_dataset[:,:,-2][..., np.newaxis], val_dataset[:,:,-1][..., np.newaxis]), \
            (test_dataset[:,:,:-2], test_dataset[:,:,-2][..., np.newaxis], test_dataset[:,:,-1][..., np.newaxis])

def split_train_test(path, test_size=0.25, seed=None):
    """Splits the subjects into separate sets for training, validation 
    and testing purposes.

    Parameters
    ----------
    path : str
        Relative or absolute path to the base/root data directory.
    test_size : float, optional
        Proportion of subjects that is used for testing and validation, by default 0.25
    seed : int, optional
        For reproducible results, define a seed, by default None

    Returns
    -------
    _, _, _ : list, list, list
        List of filenames for training, validation and testing, respectively.
    """
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
    """Gets a data generator that is callable for use with Keras.

    Parameters
    ----------
    list_files : list
        A list of filenames.
    win_len : int
        The length of a single sample, in number of samples.
    step_len : int, optional
        The number of samples that the windows slides forward, by default None

    Returns
    -------
    _ : 
        A data generator for use with tf.data.Dataset.from_generator.

    Yields
    ------
    data, labels : numpy array, numpy array
        Sensor data, labels
    """
    step_len = win_len if step_len is None else step_len

    def data_gen():
        for filename in list_files:
            with open(filename, 'rb') as infile:
                data = np.load(infile)
            for idx in range(0, data.shape[0] - win_len + 1, step_len):
                inputs = data[idx:idx+win_len,:-2]
                outputs_1 = data[idx:idx+win_len,-2][..., np.newaxis]  # gait sequences
                outputs_2 = data[idx:idx+win_len,-1][..., np.newaxis]  # gait events
                outputs_2 = tf.keras.utils.to_categorical(outputs_2, num_classes=5)
                yield inputs, {"gait_sequences": outputs_1, "gait_events": outputs_2}
    return data_gen
