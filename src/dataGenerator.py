# Generating dataset
import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, vector, labels, batch_size, n_classes, shuffle=True):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()
        self.vector = vector
        self.n_classes = n_classes

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim, 1))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            try:
                X[i, :, 0] = self.vector[ID]
                y[i] = self.params.label_to_index[self.labels[ID]]
            except Exception as e:
                X[i:, :, 0] = np.zeros([100])
                y[i] = self.params.label_to_index[self.labels[-9999]]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
