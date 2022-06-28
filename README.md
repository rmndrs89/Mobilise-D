# Mobilise-D Technical Validation Study

## Data organization
The data were organized according to the following folder structure:
```
Mobilise-D/
├── code/
├── sourcedata/
│   ├── 4002_PD/
│   │   ├── data.mat
│   │   └── infoForAlgo.mat
│   ├── 4003_PD/
│   ├── ...
│   └── 4029_CHF/
└── rawdata/
    ├── sub-4002/
    │   └── sub-4002.npy
    ├── 4003_PD/
    │   └── sub-4003.npy
    ├── ...
    └── 4029_CHF/
        └── sub-4029.npy
```

- `code/`: scripts
- `sourcedata/`: for each subject, (1) the data, organized as a nested struct in MATLAB (*.mat) format, and (2) relevant demographics information.
- `rawdata/`: for each subject, a numpy binary data file, for use with our conv neural net to detect continuous walking periods and gait events from the 6DOF raw sensor data

The `numpy` binary data file was comprised of a `numpy.ndarray` with the data from left and right foot-worn IMU, and labels for flagging **walking activity**, i.e.:
| label | activity |
|-------|-------|
|   0   | null  |
|   1   | walking   |

and specific **gait events**, where:

| label | event |
|-------|-------|
|   0   | null  |
|   1   | ICL   |
|   2   | FCL   |
|   3   | ICR   |
|   4   | FCR   |
