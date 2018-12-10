from __future__ import print_function
from src import loss, io, log
# from matplotlib import pyplot as plt
from keras.models import load_model
import os
import sys
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

parser.add_argument('-m_cpu', '--max_cpu', type=float,
                    help='Maximum number of images in RAM at once. [Default] is = 50000.00',
                    default=50000.00,
                    required=False)

parser.add_argument('-m_gpu', '--max_gpu', type=float,
                    help='Maximum number of image in GPU at once. [Default] is = 6500.00',
                    default=6500.00,
                    required=False)

parser.add_argument('-g_tile', '--grid_size', type=int,
                    help='Size of gridding tiles. [Default] is = 200',
                    default=200,
                    required=False)

parser.add_argument('-op', '--overlap', type=int,
                    help='Overlap percentage when gridding. [Default] is = 0',
                    default=0,
                    required=False)

# Parsing arguments
args = parser.parse_args()
path_data = args.data
image_size = args.size
skip_gridding = args.skip_gridding
path_model = args.model
grid_size = args.grid_size
max_num_cpu = args.max_cpu
max_num_gpu = args.max_gpu
percent_overlap = args.overlap


logger.info('percent_overlap : ' + str(percent_overlap))

start_time = time.time()

# input data
path_image = os.path.join(path_data, 'image')
logger.info('Image path is %s' % (path_image))

# Results path
path_result = os.path.join(path_data, 'result')
logger.info('Result path is %s' % (path_result))

path_tiled = os.path.join(path_result, 'tiled')

path_predict = os.path.join(path_result, 'prediction')
logger.info('Predict path is %s' % (path_predict))

# Tiled path
path_tile_image = os.path.join(path_tiled, 'image/')
logger.info('Tile image path is %s' % (path_tile_image))

# Merged Tiles path
path_merged_prediction = os.path.join(path_result, 'merged_prediction')
logger.info('Tile image path is %s' % (path_merged_prediction))

print('Tiling Images ...')
logger.info('Tiling Images..')

# Creating directory
checkdir(path_tile_image)
checkdir(path_predict)
checkdir(path_tiled)
checkdir(path_data)

if skip_gridding == 0:
    tile_image = io.checkres(path_image, grid_size,
                             path_tile_image, percent_overlap)

print('Tiling Completed')
logger.info('Tiling Completed')


# load all the training images
train_set = io.train_data()

# Definging inputs to the class
train_set.path_image = path_tile_image
train_set.path_label = path_tile_image

train_set.image_size = grid_size
train_set.max_num_cpu = max_num_cpu
train_set.max_num_gpu = max_num_gpu

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
    logger.info("shape of train_image" + str(shape_train_image))

    train_image = np.resize(train_image, [
                            shape_train_image[0], shape_train_image[1], shape_train_image[2], 3])

    # get name,size,geo referencing data
    data = io.get_geodata(train_set.image_part_list[k])

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

        # Saving data to disk
        io.write_tif(im_path, lb*255, data['geotransform']
                     [i], data['geoprojection'][i], data['size'][i])

# Merging tiled dataset to single tif
logger.info('Merging tiled dataset')
io.merge_tile(path_merged_prediction, data['name'])

logger.info('Completed')
sys.exit()
