import numpy as np
import matplotlib.pyplot as plt
import os

from utils.data_utils import SAMPLING_FREQUENCY, load_npy
from app import MyApp
from scipy.signal import find_peaks

# FILE_NAME = "/mnt/neurogeriatrics_data/MobiliseD_TVS/Free-Living/4002_PD/data.mat"
ROOT_DIR = "/mnt/neurogeriatrics_data/MobiliseD_TVS/rawdata"


def main():
    
    # Start GUI
    my_app = MyApp()
    
    return
    
if __name__ == "__main__":
    main()