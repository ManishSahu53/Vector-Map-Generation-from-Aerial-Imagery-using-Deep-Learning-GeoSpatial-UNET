from __future__ import print_function
from src import loss, io, log
# from matplotlib import pyplot as plt
from keras.models import load_model
import os
from keras.models import Model
from keras import backend as K
import cv2
import numpy as np
import time
import argparse
import logging
from src.io import checkdir
import gdal

# Setup logging
logger = log.get_logger('Testing')
log.log2file('Testing')

parser = argparse.ArgumentParser(
    description='See description below to see all available options')

parser.add_argument('-d', '--data',
                    help='Input directory containing image and label',
                    required=True)

parser.add_argument('-s', '--size', type=int,
                    help='Input size of image to be used. [Default] = 200',
                    default=200,
                    required=False)

parser.add_argument('-sg', '--skip_gridding', type=int,
                    help='If gridding is already done then skip it. [Default] is No = 0',
                    default=0,
                    required=False)

parser.add_argument('-m', '--model',
                    help='Input pre-trained model file',
                    required=True)

parser.add_argument('-t', '--tiling', type=int,
                    help='Input tiling size of image to be used. [Default] = 200',
                    default=200,
                    required=False)

# Parsing arguments
args = parser.parse_args()
path_data = args.data
image_size = args.size
skip_gridding = args.skip_gridding
path_model = args.model
tiling_size = args.tiling

percent_overlap = 0.0

start_time = time.time()

# input data
path_image = os.path.join(path_data, 'image')
logger.debug('Image path is %s' % (path_image))

# Results path
path_result = os.path.join(path_data, 'result')
logger.debug('Result path is %s' % (path_result))

path_tiled = os.path.join(path_result, 'tiled')

path_predict = os.path.join(path_result, 'prediction')
logger.debug('Predict path is %s' % (path_predict))

# Tiled path
path_tile_image = os.path.join(path_tiled, 'image/')
logger.debug('Tile image path is %s' % (path_tile_image))

print('Tiling Images ...')
logger.debug('Tiling Images..')

# Creating directory
checkdir(path_tile_image)
checkdir(path_predict)
checkdir(path_tiled)
checkdir(path_data)

if skip_gridding == 0:
    tile_image = io.checkres(path_image, tiling_size,
                             path_tile_image, percent_overlap)

print('Tiling Completed')
logger.debug('Tiling Completed')


# load all the training images
train_set = io.train_data()

# Definging inputs to the class
train_set.path_image = path_tile_image
train_set.image_size = 200
train_set.max_num_cpu = 50000.00
train_set.max_num_gpu = 6500.00

# Listing images
train_set.list_data()
part = len(train_set.image_part_list)

for k in range(part):

    # Loading the training image and labeled image
    train_image = io.get_image(
        train_set.image_part_list[k], train_set.image_size)

    shape_train_image = train_image.shape

    # Printing type and number of imgaes and labels
    print("shape of train_image" + str(shape_train_image))

    train_image = np.resize(train_image, [
                            shape_train_image[0], shape_train_image[1], shape_train_image[2], 3])

    # get name,size,geo referencing data
    data = train_set.get_data()

    # defining loss functions
    loss_ = loss.dice_coef_loss

    # loading model from model file not weights file
    model = load_model(path_model, custom_objects={
        'dice_coef_loss': loss.dice_coef_loss, 'dice_coef': loss.dice_coef, 'jaccard_coef': loss.jaccard_coef})

    # prediction model
    predict_result = model.predict(
        train_image, batch_size=16, verbose=1)  # , steps=None)

    for i in range(predict_result.shape[0]):
        # im = train_images[i]
        lb = predict_result[i, :, :, :]
        lb = np.round(lb, decimals=0)
        im_path = os.path.join(path_predict, os.path.basename(data['name'][i]))

        io.write_tif(im_path, lb*255, data['geotransform']
                     [i], data['geoprojection'][i], data['size'][i])
