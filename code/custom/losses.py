import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

class MyWeightedCategoricalCrossentropy():
    def __init__(self, weights, **kwargs):
        self.weights = weights
        super().__init__(**kwargs)
    
    def call(self, y_true, y_pred):
        loss = K.mean(K.categorical_crossentropy(y_true, y_pred) * (y_true + self.weights), axis=-1)
        return loss
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "weights": self.weights}

class MyWeightedMeanSquaredError():
    def __init__(self, weights, **kwargs):
        self.weights = weights
        super().__init__(**kwargs)
    
    def call(self, y_true, y_pred):
        loss = K.mean(K.mean(K.square(y_true - y_pred)) * (y_true + self.weights), axis=-1)
        return loss
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "weights": self.weights}