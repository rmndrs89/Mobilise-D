import tensorflow as tf
from tensorflow import keras
from tcn import TCN, tcn_full_summary
from .losses import MyWeightedCategoricalCrossentropy, MyWeightedMeanSquaredError, MyWeightedBinaryCrossentropy

def get_model(num_input_channels, **kwargs):

    # Define the model's layers
    inputs = keras.layers.Input(shape=(None, num_input_channels), name='inputs')
    hidden = TCN(**kwargs, 
        return_sequences = True,
        use_batch_norm = True,
        use_skip_connections = True,
        name = 'tcn')(inputs)
    outputs = keras.layers.Dense(units=1,
        activation = 'sigmoid',
        name = 'outputs')(hidden)

    # Define and compile the model
    model = keras.models.Model(inputs=inputs, outputs=outputs, name='gait_model')
    model.compile(
        loss = MyWeightedBinaryCrossentropy(weight=0.01),
        optimizer = keras.optimizers.Adam(learning_rate = 1e-4),
        metrics = [keras.metrics.BinaryAccuracy()]
    )
    return model

def get_multi_output_model(nb_channels, nb_classes, **kwargs):
    """
    Get a multi output model that predicts both gait sequences
    and discrete gait events from raw accelerometer and 
    gyroscope readings.

    Parameters
    ----------
    nb_channels : int
        The number of sensor channels.
    nb_classes : int
        The number of output classes (i.e., gait events).

    Returns
    -------
    model : keras.models.
        A compiled TensorFlow Keras model.
    """
    # Retrieve keyword args
    nb_filters = kwargs.get("nb_filters", 64)
    kernel_size = kwargs.get("kernel_size", 3)
    dilations = kwargs.get("dilations", [2**d for d in range(6)])
    
    # Define the layers
    inputs = keras.layers.Input(shape=(None, nb_channels), name='inputs')
    hidden = TCN(nb_filters = nb_filters,
                 kernel_size = kernel_size,
                 nb_stacks = 1,
                 dilations = dilations,
                 padding = 'same',
                 use_skip_connections = True,
                 use_batch_norm = True,
                 return_sequences = True,
                 name='tcn')(inputs)
    outputs_1 = keras.layers.Dense(units = 1,
                                   activation = 'sigmoid',
                                   name = 'gait_sequences')(hidden)
    concat = keras.layers.Concatenate(name = 'concat')([hidden, outputs_1])
    outputs_2 = keras.layers.Dense(units=nb_classes,
                                   activation = 'softmax',
                                   name = 'gait_events')(concat)
    
    # Instantiate the model
    model = keras.models.Model(inputs=inputs, outputs=[outputs_1, outputs_2], name='tcn_model')
    
    model.compile(
        loss = {'gait_sequences': MyWeightedMeanSquaredError(weight=0.01), 
                'gait_events': MyWeightedCategoricalCrossentropy(weights=[[0.1, 0.225, 0.225, 0.225, 0.225]])},
        metrics = [keras.metrics.BinaryAccuracy(), keras.metrics.CategoricalAccuracy()],
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
    )
    return model
