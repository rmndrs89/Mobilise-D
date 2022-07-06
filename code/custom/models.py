from unicodedata import name
import tensorflow as tf
from tensorflow import keras
from tcn import TCN, tcn_full_summary
from .losses import MyWeightedCategoricalCrossentropy, MyWeightedMeanSquaredError, MyWeightedBinaryCrossentropy

def get_model(num_input_channels, **kwargs):

    # Define the model's layers
    inputs = keras.layers.Input(shape=(None, num_input_channels), name='inputs')
    hidden = TCN(**kwargs,
        padding="same",
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

def get_multi_output_model(num_input_channels, threshold=None, **kwargs):
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
    # Define number of output classes
    num_output_classes = 5
    
    # Define the layers
    inputs = keras.layers.Input(shape=(None, num_input_channels), name='inputs')
    hidden = TCN(**kwargs,
                 padding = 'same',
                 use_skip_connections = True,
                 use_batch_norm = True,
                 return_sequences = True,
                 name='tcn')(inputs)
    outputs_1 = keras.layers.Dense(units = 1,
                                   activation = "sigmoid",
                                   name = "gait_sequences")(hidden)
    if threshold is not None:
        thresh = keras.layers.Lambda(lambda x: tf.where(x > threshold, 1.0, 0.0), name="thresh")(outputs_1)
        concat = keras.layers.Concatenate(name="concat")([hidden, thresh])
    else:
        concat = keras.layers.Concatenate(name = 'concat')([hidden, outputs_1])
    outputs_2 = keras.layers.Dense(units=num_output_classes,
                                   activation = 'softmax',
                                   name = 'gait_events')(concat)
    
    # Instantiate the model
    model = keras.models.Model(inputs=inputs, outputs=[outputs_1, outputs_2], name='tcn_model')
    
    # Compile the model
    model.compile(
        loss = {'gait_sequences': MyWeightedBinaryCrossentropy(weight=0.01), 
                'gait_events': MyWeightedCategoricalCrossentropy(weights=[[0.1, 0.225, 0.225, 0.225, 0.225]])},
        metrics = [keras.metrics.BinaryAccuracy(), keras.metrics.CategoricalAccuracy()],
        optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    )
    return model
