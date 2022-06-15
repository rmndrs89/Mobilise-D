```
[user@system]:~$ conda update conda
[user@system]:~$ cd ~/Projects
[user@system]:~/Projects$ mkdir Mobilise-D
[user@system]:~$ cd Mobilise-D
[user@system]:~/Projects/Mobilise-D$ conda create --prefix ./envs numpy scipy matplotlib ipykernel tensorflow
[user@system]:~/Projects/Mobilise-D$ conda activate ./envs
[user@system]:~/Projects/Mobilise-D$ conda install scikit-learn
```

Next, we check if we have access to a GPU:
```python
>>> import tensorflow as tf
>>> print(tf.__version__)
2.8.1
>>> tf.config.list_physical_devices(device_type="GPU")
2022-06-10 10:47:19.345402: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:936] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-06-10 10:47:19.399612: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:936] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-06-10 10:47:19.399740: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:936] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
