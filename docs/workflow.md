## Overview of scripts

```
code/
├── custom/
│   ├── __init__.py
│   ├── losses.py
│   └── models.py
├── train/
│   ├── final/
│   │   ├── keras_metadata.pb
│   │   ├── saved_model.pb
│   │   └── variables/
│   └── tune/
│       ├── 01/
│       │   ├── keras_metadata.pb
│       │   ├── saved_model.pb
│       │   └── variables/
│       ├── 02/
│       │   ├── keras_metadata.pb
│       │   ├── saved_model.pb
│       │   └── variables/
│       ├── ../
│       │   ├── keras_metadata.pb
│       │   ├── saved_model.pb
│       │   └── variables/
│       └── xx/
│           ├── keras_metadata.pb
│           ├── saved_model.pb
│           └── variables/
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── prepare_rawdata.py
│   └── preprocessing.py
├── batch.job
├── train.err
├── train.out
├── train.py
├── tune.err
├── tune.out
└── tune.py
```

| # | window length | # filters | kernel size | dilations | learning rate | weighting factors |
|---|---------------|-----------|-------------|-----------|---------------|-------------------|
| 01|       10      |     16    |      5      | (1, 2, 4, 8, 16, 32, 64) | 1e-4 | (0.1, 0.225, 0.225, 0.225, 0.225) |

## Overview of runs
- `train/tune/01` -> 16 conv filters, 
- `train/tune/02` -> 32 conv filters,
- `train/tune/03` -> 16 conv filters, threshold on `gait_sequences` predictions
- `train/tune/04` -> 32 conv filters, threshold on `gait_sequences` predictions
