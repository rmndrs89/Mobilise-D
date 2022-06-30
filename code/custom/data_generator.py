import numpy as np
import tensorflow as tf
import math

class MyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=32, shuffle=True, nb_classes=1):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nb_classes = nb_classes
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, idx):
        """
        Generate one batch of data.
        """
        # Get indexes of the batch
        idxs = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        # Get data (i.e., features and labels)
        features = self.data[idxs][:,:-2]
        labels = {'gait_sequences': self.data[idxs][:,-2][..., np.newaxis],
            'gait_events': tf.keras.utils.to_categorical(self.data[idxs][:,-1][..., np.newaxis],
                num_classes=self.nb_classes)}
        return features, labels


    def on_epoch_end(self):
        """
        Update indexes after each epoch.
        """
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
