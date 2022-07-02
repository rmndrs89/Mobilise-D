import os
from utils.viz_utils import plot_example
from utils.data_utils import load_npy, SAMPLING_FREQUENCY

ROOT_DIR = "/mnt/neurogeriatrics_data/MobiliseD_TVS/rawdata"

def main(filename):
    
    # Get data and labels
    data, labels = load_npy(filename)
    
    plot_example(data, labels, SAMPLING_FREQUENCY)
    return

if __name__ == "__main__":
    
    sub_id = "sub-1006"
    main(os.path.join(ROOT_DIR, sub_id, sub_id+".npy"))