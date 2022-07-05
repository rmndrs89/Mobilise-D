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
