from utils.data_utils import load_data, SAMPLING_FREQUENCY
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT_DIR = "/gxfs_work1/cau/sukne964/Mobilise-D"
WIN_LEN = int(10 * 60 * SAMPLING_FREQUENCY)

def main():
    train_data, test_data = load_data(path=ROOT_DIR, win_len=WIN_LEN)
    X_train, y1_train, y2_train = train_data

    fig, axs = plt.subplots(3, 1, sharex=True)
    for idx in range(10):
        axs[0].plot(np.arange(idx*WIN_LEN, (idx+1)*WIN_LEN), X_train[idx][:,4])
        axs[1].plot(np.arange(idx*WIN_LEN, (idx+1)*WIN_LEN), y1_train[idx][:,0])
        axs[2].plot(np.arange(idx*WIN_LEN, (idx+1)*WIN_LEN), y2_train[idx][:,0])
    plt.show()
    return

if __name__ == "__main__":
    main()
