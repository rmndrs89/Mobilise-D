from ast import Try
import tensorflow as tf
from tcn import TCN, tcn_full_summary

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