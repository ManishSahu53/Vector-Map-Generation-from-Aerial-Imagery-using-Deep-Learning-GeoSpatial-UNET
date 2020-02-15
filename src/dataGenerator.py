# Generating dataset
from tensorflow import keras
import numpy as np
import config
import cv2
import os
from src import io
import logging
from scipy import ndimage as ndi


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, imageMap, labelMap,
                 batch_size, n_classes, image_channels,
                 label_channels, image_size, prediction=False,
                 shuffle=True):
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
        self.prediction = prediction
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        if self.prediction is True:
            X = self.__data_generation(list_IDs_temp)
            return X

        elif self.prediction is False:
            X, y = self.__data_generation(list_IDs_temp)
            return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    # Reading RGB Image
    def read_image(self, path_image):
        """
        Reading image and resize it according to image size
        """
        gt, gp, size, arr = io.read_tif(path_image)
        # Taking RGB only and skipping alpha band
        arr = arr[:, :, :3]
        # print('Maximum Value Image: {}'.format(np.max(arr)))
        return cv2.resize(arr, (self.image_size, self.image_size))

    # Reading Label image
    def read_label(self, path_image):
        """
        Reading image and resize it according to image size
        """
        gt, gp, size, arr = io.read_tif(path_image)
        arr[arr == 255] = 1
        arr = np.array(arr)
        # print('Maximum Value Label: {}'.format(np.max(arr)))
        return cv2.resize(arr, (self.image_size, self.image_size))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        # print('batch_size: {}, image_size: {}, image_channels: {}'.format(
        #       self.batch_size, self.image_size, self.image_channels))
        if self.prediction is False:
            x = np.zeros((self.batch_size, self.image_size,
                          self.image_size, self.image_channels), dtype=np.float32)

            y = np.zeros((self.batch_size, self.image_size,
                          self.image_size, self.label_channels + 1), dtype=np.float32)

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Reading images and storing it
                image_arr = self.read_image(self.imageMap[ID])
                x[i] = image_arr
                y[i, :, :, 0] = self.read_label(self.labelMap[ID])

                # Creating weight matrix
                y_true_dt = ndi.distance_transform_edt(y[i, :, :, 0])
                y_true_dt[y_true_dt == 0] = -99
                # Scaling Distance Transform X 10 since values are too small
                y[i, :, :, -1] = 10/y_true_dt + 1

            return x, y

        # For prediction we dont have labeled data so Y doesn't exist
        elif self.prediction is True:
            x = np.zeros((self.batch_size, self.image_size,
                          self.image_size, self.image_channels), dtype=np.float32)

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Reading images and storing it
                image_arr = self.read_image(self.imageMap[ID])
                x[i] = image_arr

            return x


# Generating Training data map
class getData():
    def __init__(self, path_tile_image, path_tile_label):
        self.path_tiled_image = path_tile_image
        self.path_tiled_label = path_tile_label

    def getList(self):
        tiled_image = {}
        tiled_label = {}
        key = []
        index = 0

        for root, dirs, files in os.walk(self.path_tiled_image):
            for file in files:
                # If .TIF or .TIFF file found then
                if file.endswith(tuple(config.image_ext)):

                    key.append(index)
                    tiled_image[index] = os.path.join(self.path_tiled_image,
                                                      file)
                    tiled_label[index] = os.path.join(self.path_tiled_label,
                                                      file)
                    index += 1

                else:
                    continue

        return key, tiled_image, tiled_label


# Generating Testing data map
class getTestingData():
    def __init__(self, path_tile_image):
        self.path_tiled_image = path_tile_image

    def getList(self):
        tiled_image = {}
        key = []
        index = 0

        for root, dirs, files in os.walk(self.path_tiled_image):
            for file in files:
                # If .TIF or .TIFF file found then
                if file.endswith(tuple(config.image_ext)):

                    key.append(index)
                    tiled_image[index] = os.path.join(self.path_tiled_image,
                                                      file)
                    index += 1

                else:
                    continue

        if len(key) == 0:
            msg = 'Unable to get any TIF/TIFF data files in {}'.format(
                self.path_tiled_image)
            logging.error(msg)
            raise(msg)

        return key, tiled_image
