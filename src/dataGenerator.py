# Generating dataset
import keras
import numpy as np
import config
import cv2
import os


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, imageMap, labelMap,
                 batch_size, n_classes, image_channels,
                 label_channels, image_size, shuffle=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.labelMap = labelMap
        self.imageMap = imageMap
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.image_channels = image_channels
        self.label_channels = label_channels
        self.image_size = image_size
        self.shuffle = shuffle

        self.on_epoch_end()

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

    def read_image(self, path_image):
        """
        Reading image and resize it according to image size
        """
        return cv2.resize(cv2.imread(path_image), (image_size, image_size))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, self.image_size,
                      self.image_size, self.image_channels))

        y = np.empty((self.batch_size, self.image_size,
                      self.image_size, self.label_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Reading images and storing it
            x[i] = self.read_image(self.imageMap[ID])
            y[i] = self.read_image(self.labelMap[ID])

        return x, y


# Generating data map
class getData():
    def __init__(self):
        self.path_tiled_image = config.path_tiled_image
        self.path_tiled_label = config.path_tiled_label

    def getList(self):
        tiled_image = {}
        tiled_label = {}
        key = []
        index = 0

        for root, dirs, files in os.walk(self.path_tiled_image):
            for file in files:
                # If .TIF or .TIFF file found then
                if file.endwith(config.image_ext1) or \
                        file.endwith(config.image_ext2):

                    key.append(index)
                    tiled_image[index] = os.path.join(self.path_tiled_image,
                                                      file)
                    tiled_label[index] = os.path.join(self.path_tiled_label,
                                                      file)
                    index += 1

                else:
                    continue

        return key, tiled_image, tiled_label
