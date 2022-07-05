## Training

Overview of the folder structure:
```
training/
├── training/
│   ├── ... 
│   │   ├── ...
│   │   └── ...
│   ├── ...
│   └── .../
└── tuning/
    ├── README.md  # description of the runs
    └── runs/
        └── 00/
```

### Rationale
Training the model consisted of two stages:
1. **Tuning -- hyperparameter tuning** in order find the optimum model architecture. The main body of the model consisted of a so-called temporal convolutional neural network (TCN) which may consists of a different number of convolutional filters, and have different filter lengths.
2. **Training -- model trainig** to find the optimum weights that best generalize to new, unseen, data. Therefore, the model architecture from the previous step was used, and trained on the training and validation dataset. The result of the model training was used on the hold-out test data for evaluating its performance on unseen, independent data.

### Tuning
We hypothesized that the receptive field, that is the number of time steps that the model sees for predicting gait and/or gait events, is a critical paramters for it success. The receptive field is a function of the kernel size, and the number of dilations:
$$\textrm{receptive field} = 1 + 2 \cdot (\textrm{kernel size} - 1) \cdot \textrm{number of stacks} \cdot \sum\limits_{i} \textrm{dilation}_{i}$$

For the current analysis, we have frozen the `nb_stacks` as 1, therefore the receptive field only depended on the kernel size and the number of consecutive dilations. 

The tuning therefore evaluated model architecture for different number of convolutional filters, different kernel size, and different number of dilations, spanning a receptive field from minimally 1 seconds up to a maximum of 10 seconds.