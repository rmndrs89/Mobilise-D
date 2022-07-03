import numpy as np
import matplotlib.pyplot as plt
import os
from utils.data_utils import create_batch_sequences, SAMPLING_FREQUENCY

ROOT_DIR = "/mnt/neurogeriatrics_data/MobiliseD_TVS/rawdata"
# ROOT_DIR = "Z:\\MobiliseD_TVS\\rawdata"
SUB_ID = "sub-1007"

DEST_DIR = "/home/robbin/Desktop/Mobilise-D/temp"

WIN_LEN = int(6 * SAMPLING_FREQUENCY)
THR_GYRO = 10. # in degrees/s
THR_ACC = 0.5 * (1. / 9.81) # in g

def main():
    
    # Get data
    with open(os.path.join(ROOT_DIR, SUB_ID, SUB_ID+".npy"), 'rb') as infile:
        data = np.load(infile)
    
    # Get data and labels as batches of windows
    batch_sequences = np.stack(create_batch_sequences([
        os.path.join(ROOT_DIR, SUB_ID, SUB_ID+".npy")
        ],
        win_len=WIN_LEN
    ))

    # Compute the signal norms
    acc_N_left = np.linalg.norm(batch_sequences[:,:,:3], axis=-1, keepdims=True)
    om_N_left = np.linalg.norm(batch_sequences[:,:,3:6], axis=-1, keepdims=True)
    acc_N_right = np.linalg.norm(batch_sequences[:,:,6:9], axis=-1, keepdims=True)
    om_N_right = np.linalg.norm(batch_sequences[:,:,9:12], axis=-1, keepdims=True)
    
    # Compute windows standard deviations
    sig_acc_N_left = np.std(acc_N_left, axis=1)
    sig_om_N_left = np.std(om_N_left, axis=1)
    sig_acc_N_right = np.std(acc_N_right, axis=1)
    sig_om_N_right = np.std(om_N_right, axis=1)

    # Create figure
    fig, axs = plt.subplots(5, 1, sharex=True, gridspec_kw={'height_ratios': [3, 3, 3, 3, 1]})
    for idx in range(len(batch_sequences)-50, len(batch_sequences)):
        sig_acc_N_left = np.std(acc_N_left[idx], axis=0)
        sig_om_N_left = np.std(om_N_left[idx], axis=0)
        sig_acc_N_right = np.std(acc_N_right[idx], axis=0)
        sig_om_N_right = np.std(om_N_right[idx], axis=0)
        
        # Accelerations - left side        
        axs[0].plot(
            np.arange(idx*WIN_LEN, (idx+1)*WIN_LEN),
            batch_sequences[idx][:,:3],
            lw=1, ls="-"
        )
        axs[0].plot(
            [(idx+1)*WIN_LEN, (idx+1)*WIN_LEN],
            [-4, 4],
            lw=1, c=(0, 0, 0), ls=":"
        )
        axs[0].text(
            int((idx + .5) * WIN_LEN),
            3., 
            f'{sig_acc_N_left[0]:.3f}'
        )
        axs[1].plot(
            np.arange(idx*WIN_LEN, (idx+1)*WIN_LEN),
            batch_sequences[idx][:,3:6],
            lw=1, ls="-"
        )
        axs[1].plot(
            [(idx+1)*WIN_LEN, (idx+1)*WIN_LEN],
            [-400, 400],
            lw=1, c=(0, 0, 0), ls=":"
        )
        axs[1].text(
            int((idx + .5) * WIN_LEN),
            200., 
            f'{sig_om_N_left[0]:.3f}'
        )
        axs[2].plot(
            np.arange(idx*WIN_LEN, (idx+1)*WIN_LEN),
            batch_sequences[idx][:,6:9],
            lw=1, ls="-"
        )
        axs[2].plot(
            [(idx+1)*WIN_LEN, (idx+1)*WIN_LEN],
            [-4, 4],
            lw=1, c=(0, 0, 0), ls=":"
        )
        axs[2].text(
            int((idx + .5) * WIN_LEN),
            3., 
            f'{sig_acc_N_right[0]:.3f}'
        )
        axs[3].plot(
            np.arange(idx*WIN_LEN, (idx+1)*WIN_LEN),
            batch_sequences[idx][:,9:12],
            lw=1, ls="-"
        )
        axs[3].plot(
            [(idx+1)*WIN_LEN, (idx+1)*WIN_LEN],
            [-400, 400],
            lw=1, c=(0, 0, 0), ls=":"
        )
        axs[3].text(
            int((idx + .5) * WIN_LEN),
            200., 
            f'{sig_om_N_right[0]:.3f}'
        )
        axs[4].plot(
            np.arange(idx*WIN_LEN, (idx+1)*WIN_LEN),
            batch_sequences[idx][:,-2],
            lw=1, ls='-'
        )
        
    plt.show()
    return

if __name__ == "__main__":    
    main()