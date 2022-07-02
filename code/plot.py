import numpy as np
import matplotlib.pyplot as plt
import os
from utils.data_utils import create_batch_sequences, SAMPLING_FREQUENCY

# ROOT_DIR = "/mnt/neurogeriatrics_data/MobiliseD_TVS/rawdata"
ROOT_DIR = "Z:\\MobiliseD_TVS\\rawdata"
SUB_ID = "sub-1007"

WIN_LEN = int(10 * SAMPLING_FREQUENCY)

def main():
    
    # Get data and labels
    batch_sequences = np.stack(create_batch_sequences([
        os.path.join(ROOT_DIR, SUB_ID, SUB_ID+".npy")
        ],
        win_len=WIN_LEN
    ))

    acc_N = np.linalg.norm(batch_sequences[:,:,:3], axis=1)
    om_N = np.linalg.norm(batch_sequences[:,:,:3], axis=1)

    # Create figure
    fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [3, 3, 1]})
    for idx in range(25
    ):
        axs[0].plot(
            np.arange(idx*WIN_LEN, (idx+1)*WIN_LEN),
            batch_sequences[idx][:,4],
            lw=1, c="tab:blue", ls="-"
        )
        axs[0].plot(
            np.arange(idx*WIN_LEN, (idx+1)*WIN_LEN),
            om_N[:,0],
            lw=1, c='k', ls="-"
        )
        axs[1].plot(
            np.arange(idx*WIN_LEN, (idx+1)*WIN_LEN),
            batch_sequences[idx][:,10],
            lw=1, c="tab:orange", ls="-"
        )
        axs[2].plot(
            np.arange(idx*WIN_LEN, (idx+1)*WIN_LEN),
            batch_sequences[idx][:,-2],
            lw=1, c="k", ls="-"
        )
    plt.show()
    return

if __name__ == "__main__":    
    main()