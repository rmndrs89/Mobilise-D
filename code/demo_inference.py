from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import tensorflow as tf
from utils.data_utils import split_train_test, SAMPLING_FREQUENCY
from custom.losses import MyWeightedBinaryCrossentropy, MyWeightedCategoricalCrossentropy

# Define root directory
ROOT_DIR = "/mnt/neurogeriatrics_data/MobiliseD_TVS/rawdata" if sys.platform == "linux" else "Z:\\MobiliseD_TVS\\rawdata"
# ROOT_DIR = "/gxfs_work1/cau/sukne964/Mobilise-D"

def main():
    # Split subjects
    _, _, test_filenames = split_train_test(path=ROOT_DIR, test_size=0.2)
    
    # Take random test file
    idx_filename = np.random.choice(len(test_filenames), 1, replace=False)[0]
    
    # Load data
    with open(test_filenames[idx_filename], 'rb') as infile:
        data = np.load(infile)
    
    # Split features and labels
    features, labels = data[:,:-2], data[:,-2:]
    
    # Get trained model
    trained_model = tf.keras.models.load_model(
        "./train/tune/01", custom_objects={
            "MyWeightedBinaryCrossentropy": MyWeightedBinaryCrossentropy(0.01),
            "MyWeightedCategoricalCrossentropy": MyWeightedCategoricalCrossentropy([[0.1, 0.225, 0.225, 0.225, 0.225]])
        }
    )
    
    # Make predictions
    predictions = trained_model.predict(features[np.newaxis, ...])
    
    # Visualize
    fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
    
    # Input data -- gyroscope signals left and right
    axs[0].plot(np.arange(len(features))/SAMPLING_FREQUENCY, features[:,4], label="gyr_L")
    axs[0].plot(np.argwhere(labels[:,1]==1)[:,0]/SAMPLING_FREQUENCY, features[np.argwhere(labels[:,1]==1)[:,0],4], ls="none", marker="o", mfc="tab:blue", mec="tab:blue", ms=4)
    axs[0].plot(np.argwhere(labels[:,1]==2)[:,0]/SAMPLING_FREQUENCY, features[np.argwhere(labels[:,1]==2)[:,0],4], ls="none", marker="o", mfc="tab:blue", mec="none", ms=4)
    axs[0].plot(np.arange(len(features))/SAMPLING_FREQUENCY, features[:,10], label="gyr_R")
    
    # Gait sequences -- annotated and predicted
    axs[1].fill_between(np.arange(len(labels))/SAMPLING_FREQUENCY, labels[:,0], fc="tab:green", alpha=0.2, ec="none", label="annotated")
    axs[1].plot(np.arange(len(predictions[0][0]))/SAMPLING_FREQUENCY, predictions[0][0][:,0], lw=1, c="tab:green", label="predicted")
    
    # Gait events -- annotated and predicted
    axs[2].plot(np.arange(len(predictions[1][0]))/SAMPLING_FREQUENCY, predictions[1][0][:,1], lw=1, c="tab:blue", label=r"IC${_L}$")
    axs[2].plot(np.arange(len(predictions[1][0]))/SAMPLING_FREQUENCY, predictions[1][0][:,2], lw=1, c="tab:orange", label=r"FC${_L}$")
    axs[2].plot(np.arange(len(predictions[1][0]))/SAMPLING_FREQUENCY, predictions[1][0][:,3], lw=1, c="tab:green", label=r"IC$_{R}$")
    axs[2].plot(np.arange(len(predictions[1][0]))/SAMPLING_FREQUENCY, predictions[1][0][:,4], lw=1, c="tab:red", label=r"FC$_{R}$")
    
    # Labels, ticks, et cetera
    axs[0].set_ylabel(r"angular velocity $\omega(t)$, in $^{\circ}$ s$^{-1}$")
    axs[0].legend()
    axs[1].set_ylim([-0.1, 1.1])
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(["", "walking"])
    axs[1].legend()
    axs[2].set_xlabel(r"time $t$, in s")
    axs[2].set_ylabel(r"$\mathrm{Pr}(\mathbf{x}(t_{n})=y_{i})$")
    axs[2].set_ylim([-0.1, 1.1])
    axs[2].set_yticks([0, 1])
    axs[2].legend()
    plt.show()
    return 

if __name__ == "__main__":
    main()