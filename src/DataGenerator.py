import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf

class CSVDataGenerator(Sequence):
    def __init__(self, csv_file, batch_size, n_classes, shuffle=True):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data = pd.read_csv(csv_file)
        self.indexes = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.data.iloc[batch_indexes]

        X = batch_data.iloc[:, :-1].values
        y = batch_data.iloc[:, -1].values

        y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)