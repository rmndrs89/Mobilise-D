import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

def plot_example(data, labels, fs):
    
    fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [3, 3, 1]})
    axs[0].fill_between(np.arange(len(labels)), labels[:,0]*data[:,4].max(), fc='tab:green', alpha=0.05, ls=':')
    axs[0].fill_between(np.arange(len(labels)), labels[:,0]*data[:,4].min(), fc='tab:green', alpha=0.05, ls=':')
    axs[0].plot(np.arange(len(data)), data[:,10], ls='-', c='tab:orange', alpha=0.1)
    axs[0].plot(np.arange(len(data)), data[:,4], ls='-', c='tab:blue')
    axs[0].plot(np.argwhere(labels[:,1]==1)[:,0], data[np.argwhere(labels[:,1]==1)[:,0],4], ls='none', marker='o', mfc='tab:blue', mec='tab:blue')
    axs[0].plot(np.argwhere(labels[:,1]==2)[:,0], data[np.argwhere(labels[:,1]==2)[:,0],4], ls='none', marker='o', mfc='none', mec='tab:blue')
    axs[1].fill_between(np.arange(len(labels)), labels[:,0]*data[:,10].max(), fc='tab:green', alpha=0.05, ls=':')
    axs[1].fill_between(np.arange(len(labels)), labels[:,0]*data[:,10].min(), fc='tab:green', alpha=0.05, ls=':')
    axs[1].plot(np.arange(len(data)), data[:,4], ls='-', c='tab:blue', alpha=0.1)
    axs[1].plot(np.arange(len(data)), data[:,10], ls='-', c='tab:orange')
    axs[1].plot(np.argwhere(labels[:,1]==3)[:,0], data[np.argwhere(labels[:,1]==3)[:,0],10], ls='none', marker='o', mfc='tab:orange', mec='tab:orange')
    axs[1].plot(np.argwhere(labels[:,1]==4)[:,0], data[np.argwhere(labels[:,1]==4)[:,0],10], ls='none', marker='o', mfc='none', mec='tab:orange')
    axs[2].fill_between(np.arange(len(labels)), labels[:,0], fc='tab:green', alpha=0.5, ls=':')
    axs[2].plot(np.arange(len(labels)), labels[:,0], ls='-', c='k', lw=1)
    
    axs[0].grid(True, ls=':', c=(0, 0, 0), alpha=0.5)
    axs[1].grid(True, ls=':', c=(0, 0, 0), alpha=0.5)
    
    axs[2].set_ylim([-.1, 1.1])
    plt.show()
    return