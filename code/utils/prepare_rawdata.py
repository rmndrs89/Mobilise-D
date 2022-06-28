import os
from re import sub

from data_utils import load_mat
import numpy as np
from preprocessing import _resample
import matplotlib.pyplot as plt

ROOT_DIR = "/mnt/neurogeriatrics_data/MobiliseD_TVS/sourcedata"
DEST_DIR = "/mnt/neurogeriatrics_data/MobiliseD_TVS/rawdata"

SAMPLING_FREQUENCY = 64.0 # Hz

def parse_file(file_name, dest_dir="", visualize=False):
    
    # Retrieve subject id
    _, sub_id = os.path.split(os.path.split(file_name)[0])
    
    # Load raw data
    raw = load_mat(file_name)
    
    # Preliminary checks
    if "TimeMeasure1" not in list(raw["data"].keys()):
        print(f"{file_name:s}")
        return
    if "Recording4" not in list(raw["data"]["TimeMeasure1"].keys()):
        return
    if "SU_INDIP" not in list(raw["data"]["TimeMeasure1"]["Recording4"].keys()) or \
        "Standards" not in list(raw["data"]["TimeMeasure1"]["Recording4"].keys()):
            return
    if "INDIP" not in list(raw['data']['TimeMeasure1']['Recording4']['Standards'].keys()):
        print(f"INDIP not available for {file_name:s}")
        return
    if "LeftFoot" not in list(raw["data"]["TimeMeasure1"]["Recording4"]["SU_INDIP"].keys()):
        print(f"{file_name:s} does not contain data for the left foot IMU.")
        return
    elif "RightFoot"  not in list(raw["data"]["TimeMeasure1"]["Recording4"]["SU_INDIP"].keys()):
        print(f"{file_name:s} does not contain data for the right foot IMU.")
        return
    
    # Check sampling frequencies
    f_s = float(raw["data"]["TimeMeasure1"]["Recording4"]["SU_INDIP"]["LeftFoot"]["Fs"]["Acc"])
    if float(raw["data"]["TimeMeasure1"]["Recording4"]["SU_INDIP"]["LeftFoot"]["Fs"]["Gyr"]) != f_s:
        print(f"{file_name:s} > sampling frequency does not match.")
        return
    if float(raw["data"]["TimeMeasure1"]["Recording4"]["SU_INDIP"]["RightFoot"]["Fs"]["Acc"]) != f_s:
        print(f"{file_name:s} > sampling frequency does not match.")
        return
    if float(raw["data"]["TimeMeasure1"]["Recording4"]["SU_INDIP"]["RightFoot"]["Fs"]["Gyr"]) != f_s:
        print(f"{file_name:s} > sampling frequency does not match.")
        return
    
    # Get all relevant data
    data = raw["data"]["TimeMeasure1"]["Recording4"]["SU_INDIP"]["LeftFoot"]["Acc"]
    data = np.hstack((data, raw["data"]["TimeMeasure1"]["Recording4"]["SU_INDIP"]["LeftFoot"]["Gyr"]))
    data = np.hstack((data, raw["data"]["TimeMeasure1"]["Recording4"]["SU_INDIP"]["RightFoot"]["Acc"]))
    data = np.hstack((data, raw["data"]["TimeMeasure1"]["Recording4"]["SU_INDIP"]["RightFoot"]["Gyr"]))
    
    # Resample data
    data = _resample(data, fs_old=f_s, fs_new=SAMPLING_FREQUENCY)
    
    # Get dimensions
    num_time_steps, num_channels = data.shape
    
    # Loop over the continuous walking periods
    activity_labels = np.zeros((num_time_steps, 1))  # pre-allocate binary vector to flag walking activity
    event_labels = np.zeros((num_time_steps,1))  # pre-allocate vector to label gait events
    for idx in range(len(raw["data"]["TimeMeasure1"]["Recording4"]["Standards"]["INDIP"]["ContinuousWalkingPeriod"])):
        
        # Get continuous walking period
        start_time = raw['data']['TimeMeasure1']['Recording4']['Standards']['INDIP']['ContinuousWalkingPeriod'][idx]['Start']
        end_time = raw['data']['TimeMeasure1']['Recording4']['Standards']['INDIP']['ContinuousWalkingPeriod'][idx]['End']
        idx_start = int(start_time * SAMPLING_FREQUENCY)
        idx_end = int(end_time * SAMPLING_FREQUENCY)
        activity_labels[idx_start:idx_end+1,:] = 1
        
        # Get indices of initial contacts
        initial_contact_times = raw['data']['TimeMeasure1']['Recording4']['Standards']['INDIP']['ContinuousWalkingPeriod'][idx]['InitialContact_Event']
        for jdx in range(len(initial_contact_times)):
            if not np.isnan(initial_contact_times[jdx]):
                _idx = int(initial_contact_times[jdx] * SAMPLING_FREQUENCY)
                if raw['data']['TimeMeasure1']['Recording4']['Standards']['INDIP']['ContinuousWalkingPeriod'][idx]['InitialContact_LeftRight'][jdx] == 'Left':
                    event_labels[_idx,:] = 1
                elif raw['data']['TimeMeasure1']['Recording4']['Standards']['INDIP']['ContinuousWalkingPeriod'][idx]['InitialContact_LeftRight'][jdx] == 'Right':
                    event_labels[_idx,:] = 3
        
        # Likewise for final contacts
        final_contact_times = raw['data']['TimeMeasure1']['Recording4']['Standards']['INDIP']['ContinuousWalkingPeriod'][idx]['FinalContact_Event']
        for jdx in range(len(final_contact_times)):
            if not np.isnan(final_contact_times[jdx]):
                _idx = int(final_contact_times[jdx] * SAMPLING_FREQUENCY)
                if raw['data']['TimeMeasure1']['Recording4']['Standards']['INDIP']['ContinuousWalkingPeriod'][idx]['FinalContact_LeftRight'][jdx] == 'Left':
                    event_labels[_idx,:] = 2
                elif raw['data']['TimeMeasure1']['Recording4']['Standards']['INDIP']['ContinuousWalkingPeriod'][idx]['FinalContact_LeftRight'][jdx] == 'Right':
                    event_labels[_idx,:] = 4
    
    # Plot bouts and events
    if visualize:
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].fill_between(np.arange(data.shape[0]), activity_labels[:,0]*data[:,4].max(), color=(0, 1, 0), alpha=0.1)
        axs[0].fill_between(np.arange(data.shape[0]), activity_labels[:,0]*data[:,4].min(), color=(0, 1, 0), alpha=0.1)
        axs[0].plot(data[:,4], c="tab:blue", lw=1)
        axs[0].plot(np.argwhere(event_labels[:,0]==2)[:,0], data[np.argwhere(event_labels[:,0]==2)[:,0],4], c="tab:blue", ls='none', marker='^', mfc='none', mec='tab:blue')
        axs[0].plot(np.argwhere(event_labels[:,0]==1)[:,0], data[np.argwhere(event_labels[:,0]==1)[:,0],4], c="tab:blue", ls='none', marker='v', mfc='none', mec='tab:blue')
        axs[0].grid(True, c=(0, 0, 0), alpha=0.2, ls=":")
        axs[1].fill_between(np.arange(data.shape[0]), activity_labels[:,0]*data[:,10].max(), color=(0, 1, 0), alpha=0.1)
        axs[1].fill_between(np.arange(data.shape[0]), activity_labels[:,0]*data[:,10].min(), color=(0, 1, 0), alpha=0.1)
        axs[1].plot(data[:,10], c="tab:orange", lw=1)
        axs[1].plot(np.argwhere(event_labels[:,0]==4)[:,0], data[np.argwhere(event_labels[:,0]==4)[:,0],10], c="tab:orange", ls='none', marker='^', mfc='none', mec='tab:orange')
        axs[1].plot(np.argwhere(event_labels[:,0]==3)[:,0], data[np.argwhere(event_labels[:,0]==3)[:,0],10], c="tab:orange", ls='none', marker='v', mfc='none', mec='tab:orange')
        axs[1].grid(True, c=(0, 0, 0), alpha=0.2, ls=":")
        axs[1].set_xlabel("time / s")
        plt.show()
        
    # Save data and labels
    if not os.path.isdir(os.path.join(DEST_DIR, "sub-"+sub_id[:4])):
        os.mkdir(os.path.join(DEST_DIR, "sub-"+sub_id[:4]))
    if not os.path.isfile(os.path.join(DEST_DIR, "sub-"+sub_id[:4], "sub-"+sub_id[:4]+".npy")):    
        with open(os.path.join(DEST_DIR, "sub-"+sub_id[:4], "sub-"+sub_id[:4]+".npy"), 'wb') as outfile:
            np.save(outfile, np.hstack((data, activity_labels, event_labels)))
        print(f"Data saved to ... {os.path.join(DEST_DIR, 'sub-'+sub_id[:4], 'sub-'+sub_id[:4]+'.npy'):s}")
    else:
        print(f"Data already exists for ... {os.path.join(DEST_DIR, 'sub-'+sub_id[:4], 'sub-'+sub_id[:4]+'.npy'):s}")
    return

def prepare_rawdata(root_dir, dest_dir=""):
    """
    MATLAB function to prepare the dataset in an organized folder structure.
    
    Given the source directory, the root directory with the original MATLAB *.mat files,
    the corresponding `data.mat` file is opened.
    
    Then, the accelerometer and gyroscope data from the left and right IMU data are read,
    the continuous walking periods, and the detected initial and final contacts are read.
    """
    if dest_dir == "":
        dest_dir = root_dir
    
    # Get list of subject ids
    sub_ids = [sub_id for sub_id in os.listdir(root_dir) if sub_id.startswith("50")]
    
    # Loop over the subject ids
    for (ix_sub_id, sub_id) in enumerate(sub_ids):
        print(f"{sub_id}")
        if os.path.isfile(os.path.join(root_dir, sub_id, "data.mat")):
            parse_file(os.path.join(root_dir, sub_id, "data.mat"), dest_dir=dest_dir, visualize=False)
    return

if __name__ == "__main__":
    prepare_rawdata(root_dir=ROOT_DIR, dest_dir=DEST_DIR)