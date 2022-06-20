from tkinter.tix import Tree
from matplotlib import units
import tensorflow as tf
from tensorflow import keras
from tcn import TCN, tcn_full_summary

def get_multi_output_model(nb_channels, nb_classes):
    
    # Define the layers
    inputs = keras.layers.Input(shape=(None, nb_channels), name='inputs')
    hidden = TCN(nb_filters=64,
                 kernel_size=3,
                 nb_stacks=1,
                 dilations=[2**d for d in range(7)],
                 padding='same',
                 use_skip_connections=True,
                 use_batch_norm=True,
                 return_sequences=True,
                 name='tcn')(inputs)
    outputs_1 = keras.layers.Dense(units=1,
                                   activation='sigmoid',
                                   name='gait_sequences')(hidden)
    concat = keras.layers.Concatenate(name='concat')([hidden, outputs_1])
    outputs_2 = keras.layers.Dense(units=nb_classes,
                                   activation='softmax',
                                   name='gait_events')(concat)
    
    # Instantiate the model
    model = keras.models.Model(inputs=inputs, outputs=[outputs_1, outputs_2], name='tcn_model')
    
    # Compile the model
    model.compile(
        loss = [keras.losses.BinaryCrossentropy(), keras.losses.CategoricalCrossentropy()],
        metrics = [keras.metrics.BinaryAccuracy(), keras.metrics.CategoricalAccuracy()],
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
    )
    return model

def build_model():
    
    # Define model architecture
    inputs = tf.keras.layers.Input(shape=(None, 12), name='inputs')
    tcn = TCN(
        nb_filters=32,
        kernel_size=3,
        nb_stacks=1,
        dilations=[2**d for d in range(7)],
        padding='same',
        use_skip_connections=True,
        use_batch_norm=True,
        return_sequences=True,
        name='tcn'
    )(inputs)
    outputs = tf.keras.layers.Dense(
        units=1, 
        activation='sigmoid',
        name='outputs'
    )(tcn)
    
    # Instantiate model
    model = tf.keras.models.Model(
        inputs=inputs, outputs=outputs, name='tcn_model'
    )
    
    # Compile
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['acc']
    )
    return model