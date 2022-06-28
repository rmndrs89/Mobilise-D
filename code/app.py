import sys
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
from utils.data_utils import SAMPLING_FREQUENCY, load_npy

class MyApp:
    def __init__(self, file_name="", init_dir="", root=None, cols=[4, 10]):
        
        # Get inputs
        self.file_name = file_name
        if self.file_name != "":
            self.data, self.labels = load_npy(self.file_name)
        else:
            self.data, self.labels = [], []
        self.init_dir = init_dir
        if self.init_dir == "":
            if sys.platform == "linux":
                self.init_dir = "/mnt/neurogeriatrics_data/MobiliseD_TVS/rawdata"
            else:
                self.init_dir = "Z:\\MobiliseD_TVS\\rawdata"
        self._cols = cols
                
        # Initialize GUI
        if root is not None:
            self.root = root
        else:
            self.root = tk.Tk()
        self.root.title("Mobilise-D Annotation GUI")
        self._set_layout()        
                        
        # Infinite event loop        
        self.root.mainloop()
    
    def _set_layout(self):
        
        # Set initial size
        self.root.geometry("1024x800")
        
        # Responsive window
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        
        # Top frame
        self.frm_top = tk.Frame(master=self.root)
        self.frm_top.columnconfigure([0, 2], weight=0)
        self.frm_top.columnconfigure(1, weight=1)
        
        self.btn_browse = tk.Button(master=self.frm_top, text="Browse", command=self._browse_callback)
        self.ent_file_name = tk.Entry(master=self.frm_top)
        self.ent_file_name.insert(0, self.file_name)
        self.btn_load = tk.Button(master=self.frm_top, text="Load", command=self._load_callback)
        
        self.btn_browse.grid(row=0, column=0, sticky="nw")
        self.ent_file_name.grid(row=0, column=1, sticky="new")
        self.btn_load.grid(row=0, column=2, sticky="ne")
        self.frm_top.grid(row=0, column=0, sticky="new", padx=5, pady=5)
        
        # Main frame
        self.frm_main = tk.Frame(master=self.root)
        
        self.fig, self.axs = plt.subplots(2, 1, sharex=True, sharey=True)
        if len(self.data) > 0: # if not an empty list
            self.axs[0].plot(
                np.arange(self.data.shape[0]),
                self.data[:,self._cols[1]],
                ls='-', c="tab:orange", alpha=0.3, lw=1
            )
            self.axs[0].plot(
                np.arange(self.data.shape[0]),
                self.data[:,self._cols[0]],
                ls='-', c="tab:blue", lw=1
            )
            self.axs[0].grid(True, c=(0, 0, 0), alpha=0.1, ls=":")
        else:
            self.axs[0].plot()
            self.axs[1].plot()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frm_main)        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frm_main)
        self.toolbar.pack(side=tk.TOP, fill=tk.X, expand=0)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect("button_press_event", self._onclick_callback)
        self.canvas.draw()
        
        self.frm_main.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        return
    
    def _browse_callback(self):
        
        # Prompt user
        filepath = askopenfilename(
            initialdir=self.init_dir,
            filetypes=[("Python Files", "*.npy"), 
                       ("All Files", "*.*")]
        )
        
        # If not given, return empty-handed
        if not filepath:
            return
        
        # Set object filename
        self.file_name = filepath
        
        # Update Entry widget
        self.ent_file_name.delete(0, tk.END)
        self.ent_file_name.insert(0, self.file_name)
        return

    def _load_callback(self):
        if self.file_name != "":
            # Delete any previous data
            self.data, self.labels = [], []
            
            # Load data
            self.data, self.labels = load_npy(self.file_name)
            
            # Get indices of local minima
            self._get_minima()
            
        # Update plot
        self._update_figure()
        return

    def _onclick_callback(self, event):
        
        # If not left or right mouse button click, or if we are in zoom/pan mode
        if event.button not in [1, 3] or str(self.canvas.toolbar.mode):
            return
        
        # Annotate local minima as potential final contacts (toe-off) by clicking
        if event.inaxes == self.axs[0]:

            # Find local minimum closest to where the user clicked
            idx = np.argmin(np.abs(self.idx_min_LF - event.xdata))
            
            self.axs[0].plot(
                self.idx_min_LF[idx],
                self.data[self.idx_min_LF[idx],self._cols[0]],
                ls="-", marker="o", mfc="tab:blue", mec="tab:blue", ms=6
            )
        elif event.inaxes == self.axs[1]:
            idx = np.argmin(np.abs(self.idx_min_RF - event.xdata))
            self.axs[1].plot(
                self.idx_min_RF[idx],
                self.data[self.idx_min_RF[idx],self._cols[1]],
                ls="-", marker="o", mfc="tab:orange", mec="tab:orange", ms=6
            )
        self.canvas.draw()
        return
        
    def _get_minima(self):
        """
        Get indices of local minima as suggestions for the final contacts.
        Do not show these local minima until user clicks in the axes.
        """
        from scipy.signal import find_peaks
        import numpy as np
        
        # Get indices of negative peaks
        self.idx_min_LF, _ = find_peaks(
            -self.data[:,self._cols[0]],
            distance=int(SAMPLING_FREQUENCY/4),
            height=0.0
        )
        self.idx_min_RF, _ = find_peaks(
            -self.data[:,self._cols[1]],
            distance=int(SAMPLING_FREQUENCY/4),
            height=0.0
        )
        return
    
    def _update_figure(self):
        def _format_func(value, tick_number):
            return f"{value/SAMPLING_FREQUENCY}"

        self.axs[0].clear()
        self.axs[1].clear()
        if self.data is not None:
            self.axs[0].fill_between(
                np.arange(self.data.shape[0]),
                self.labels[:,-2] * self.data[:,self._cols].max().max(),
                fc=(1, 0, 1), alpha=0.1, ec='none'
            )
            self.axs[0].fill_between(
                np.arange(self.data.shape[0]),
                self.labels[:,-2] * self.data[:,self._cols].min().min(),
                fc=(1, 0, 1), alpha=0.1, ec='none'
            )
            self.axs[0].plot(
                self.data[:,self._cols[1]],
                ls="-", c="tab:orange", alpha=0.2, lw=0.5
            )
            self.axs[0].plot(
                self.data[:,self._cols[0]],
                ls="-", c="tab:blue", lw=1
            )
            self.axs[0].plot(
                np.argwhere(self.labels[:,-1]==2)[:,0],
                self.data[np.argwhere(self.labels[:,-1]==2)[:,0],self._cols[0]],
                ls='none', marker='^', mfc='none', mec="tab:blue", ms=6,
                label="FC"
            )
            self.axs[0].plot(
                np.argwhere(self.labels[:,-1]==1)[:,0],
                self.data[np.argwhere(self.labels[:,-1]==1)[:,0],self._cols[0]],
                ls='none', marker='v', mfc='none', mec="tab:blue", ms=6,
                label="IC"
            )
            
            self.axs[1].fill_between(
                np.arange(self.data.shape[0]),
                self.labels[:,-2] * self.data[:,self._cols].max().max(),
                fc=(1, 0, 1), alpha=0.1, ec='none'
            )
            self.axs[1].fill_between(
                np.arange(self.data.shape[0]),
                self.labels[:,-2] * self.data[:,self._cols].min().min(),
                fc=(1, 0, 1), alpha=0.1, ec='none'
            )
            self.axs[1].plot(
                self.data[:,self._cols[0]],
                ls="-", c="tab:blue", alpha=0.2, lw=0.5
            )
            self.axs[1].plot(
                self.data[:,self._cols[1]],
                ls="-", c="tab:orange", lw=1
            )
            self.axs[1].plot(
                np.argwhere(self.labels[:,-1]==4)[:,0],
                self.data[np.argwhere(self.labels[:,-1]==4)[:,0],self._cols[1]],
                ls='none', marker='^', mfc='none', mec="tab:orange", ms=6,
                label="FC"
            )
            self.axs[1].plot(
                np.argwhere(self.labels[:,-1]==3)[:,0],
                self.data[np.argwhere(self.labels[:,-1]==3)[:,0],self._cols[1]],
                ls='none', marker='v', mfc='none', mec="tab:orange", ms=6,
                label="IC"
            )

            # https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html
            self.axs[1].xaxis.set_major_formatter(plt.FuncFormatter(_format_func))
            self.canvas.draw()
        return