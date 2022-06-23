import tensorflow as tf
from tensorflow import keras
from tcn import TCN
import keras_tuner as kt

class TCNHyperModel(kt.HyperModel):
    def __init__(self, nb_channels):
        self.nb_channels = nb_channels
    
    def build(self, hp):
        # Define the layers
        inputs = keras.layers.Input(shape=(None, self.nb_channels), name="inputs")
        tcn = TCN(
            nb_filters = 2**hp.Int("nb_filters", min_value=4, max_value=6, step=1),
            return_sequences = True,
            name = "tcn"
        )(inputs)
        outputs = keras.layers.Dense(
            units = 1,
            activation = "sigmoid", 
            name = "outputs"
        )(tcn)

        # Instantiate the model
        model = keras.models.Model(inputs=inputs, outputs=outputs, name="tcn_model")

        # Compile the model
        model.compile(
            loss = "binary_crossentropy",
            optimizer = keras.optimizers.Adam(learning_rate=0.001),
            metrics = ["binary_accuracy"]
        )
        return model
