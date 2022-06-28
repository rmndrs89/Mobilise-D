import tensorflow as tf
from tensorflow import keras
from tcn import TCN
import keras_tuner as kt
from .losses import MyWeightedMeanSquaredError, MyWeightedCategoricalCrossentropy

class TCNHyperModel(kt.HyperModel):
    def __init__(self, nb_channels, nb_classes, weights_1, weights_2):
        self.nb_channels = nb_channels
        self.nb_classes = nb_classes
        self.weights_1 = weights_1
        self.weights_2 = weights_2
    
    def build(self, hp):
        # Define the layers
        inputs = keras.layers.Input(shape=(None, self.nb_channels), name="inputs")
        tcn = TCN(
            nb_filters = 2**hp.Int("nb_filters", min_value=4, max_value=6, step=1),
            dilations = [2**d for d in range(hp.Int("dilations", min_value=2, max_value=6, step=1))],
            dropout_rate = hp.Choice("dropout_rate", [0.0, 0.05]),
            return_sequences = True,
            name = "tcn"
        )(inputs)
        outputs_1 = keras.layers.Dense(
            units = 1,
            activation = "sigmoid", 
            name = "gait_sequences"
        )(tcn)
        concat = keras.layers.Concatenate(name="concat")([tcn, outputs_1])
        outputs_2 = keras.layers.Dense(
            units = self.nb_classes,
            activation = "softmax",
            name = "gait_events"
        )(concat)

        # Instantiate the model
        model = keras.models.Model(inputs=inputs, outputs=[outputs_1, outputs_2], name="tcn_multi_output_model")

        # Compile the model
        model.compile(
            loss = {"gait_sequences": MyWeightedMeanSquaredError(self.weights_1),
                    "gait_events": MyWeightedCategoricalCrossentropy(self.weights_2)},
            optimizer = keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")),
            metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.CategoricalCrossentropy()]
        )
        return model
