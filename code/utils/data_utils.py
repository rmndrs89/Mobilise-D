import numpy as np
import scipy.io

SAMPLING_FREQUENCY = 64. # Hz

def load_npy(file_name):
    """
    Load the data from the custom .npy file.
    The data columns correspond to the accelerometer and gyroscope
    data of the left and right foot, respectively.
    
    The last two columns represented the labels, 
    with the one-but-last column corresponding to continuous walking periods,
    and the last column corresponding to specific gait events.

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

# def get_data(data_dict, tracked_points=[]):
#     """
#     Get data and labels from the data dictionary.

#     Parameters
#     ----------
#     data_dict : dict
#         Data from the wearable sensor system organized as a nested dictionary.
#     tracked_point : list
#         List of tracked points from which to use the data for classifying and predicting.
#     """
#     if not isinstance(tracked_points, list):
#         tracked_points = [tracked_points]
#     if len(tracked_points) == 0:
#         return
    
#     for idx_tracked_point, tracked_point in enumerate(tracked_points):
#         if idx_tracked_point > 0:
#             acc_data = np.hstack((acc_data, data_dict["data"]["TimeMeasure1"]["Recording4"]["SU_INDIP"][tracked_point]["Acc"]))
#             gyr_data = np.hstack((gyr_data, data_dict["data"]["TimeMeasure1"]["Recording4"]["SU_INDIP"][tracked_point]["Gyr"]))
#         else:
#             acc_data = data_dict["data"]["TimeMeasure1"]["Recording4"]["SU_INDIP"][tracked_point]["Acc"]
#             gyr_data = data_dict["data"]["TimeMeasure1"]["Recording4"]["SU_INDIP"][tracked_point]["Gyr"]
#     return acc_data, gyr_data

# def get_labels(data_dict, type="events"):
    
#     # Classify just gait, ...
#     if type == "gait":
        
#         # Get data to preallocate zeros array
#         acc_data, _ = get_data(data_dict=data_dict, tracked_points=["LeftFoot"])
        
#         gt_labels = np.zeros((acc_data.shape[0], 1))
#         for idx in range(len(data_dict["data"]["TimeMeasure1"]["Recording4"]["Standards"]["INDIP"]["ContinuousWalkingPeriod"])):
#             idx_start = int(data_dict["data"]["TimeMeasure1"]["Recording4"]["Standards"]["INDIP"]["ContinuousWalkingPeriod"][idx]["Start"] * SAMPLING_FREQUENCY)
#             idx_end = int(data_dict["data"]["TimeMeasure1"]["Recording4"]["Standards"]["INDIP"]["ContinuousWalkingPeriod"][idx]["End"] * SAMPLING_FREQUENCY)
#             gt_labels[idx_start-1:idx_end,:] = 1
#         return gt_labels
    
#     # ..., or each individual gait event
#     else:
        
#         # Get reference timings of initial contacts
#         ts_ICL, ts_ICR = [], []
#         for idx in range(len(data_dict["data"]["TimeMeasure1"]["Recording4"]["Standards"]["INDIP"]["ContinuousWalkingPeriod"])):
#             ts = data_dict["data"]["TimeMeasure1"]["Recording4"]["Standards"]["INDIP"]["ContinuousWalkingPeriod"][idx]["InitialContact_Event"]
#             indices = [ind for ind, elem in enumerate(data_dict["data"]["TimeMeasure1"]["Recording4"]["Standards"]["INDIP"]["ContinuousWalkingPeriod"][idx]["InitialContact_LeftRight"]) if elem == "Left"]
#             ts_ICL += ts[indices].tolist()
#             del indices
#             indices = [ind for ind, elem in enumerate(data_dict["data"]["TimeMeasure1"]["Recording4"]["Standards"]["INDIP"]["ContinuousWalkingPeriod"][idx]["InitialContact_LeftRight"]) if elem == "Right"]
#             ts_ICR += ts[indices].tolist()
#             del indices, ts
            
#         # Likewise for final contacts
#         ts_FCL, ts_FCR = [], []
#         for idx in range(len(data_dict["data"]["TimeMeasure1"]["Recording4"]["Standards"]["INDIP"]["ContinuousWalkingPeriod"])):
#             ts = data_dict["data"]["TimeMeasure1"]["Recording4"]["Standards"]["INDIP"]["ContinuousWalkingPeriod"][idx]["FinalContact_Event"]
#             indices = [ind for ind, elem in enumerate(data_dict["data"]["TimeMeasure1"]["Recording4"]["Standards"]["INDIP"]["ContinuousWalkingPeriod"][idx]["FinalContact_LeftRight"]) if elem == "Left"]
#             ts_FCL += ts[indices].tolist()
#             del indices
#             indices = [ind for ind, elem in enumerate(data_dict["data"]["TimeMeasure1"]["Recording4"]["Standards"]["INDIP"]["ContinuousWalkingPeriod"][idx]["FinalContact_LeftRight"]) if elem == "Right"]
#             ts_FCR += ts[indices].tolist()
#             del indices, ts
        
#         # Return dictionary of gait events
#         return {"ICL": np.array([int(t * SAMPLING_FREQUENCY) for t in ts_ICL if not np.isnan(t)]), 
#                 "ICR": np.array([int(t * SAMPLING_FREQUENCY) for t in ts_ICR if not np.isnan(t)]),
#                 "FCL": np.array([int(t * SAMPLING_FREQUENCY) for t in ts_FCL if not np.isnan(t)]),
#                 "FCR": np.array([int(t * SAMPLING_FREQUENCY) for t in ts_FCR if not np.isnan(t)])}
        
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
        