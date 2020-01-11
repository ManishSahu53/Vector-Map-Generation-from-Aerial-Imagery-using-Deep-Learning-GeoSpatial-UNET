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
from src import postprocess
from src import metric, io, util

import gdal


# Setup logging
logger = log.get_logger('evaluating')
log.log2file('evaluating')

parser = argparse.ArgumentParser(
    description='See description below to see all available options')

parser.add_argument('-d', '--data',
                    help='Input directory containing image and label',
                    required=True)

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

parser.add_argument('-gs', '--grid_size', type=int,
                    help='Size of gridding tiles. [Default] is = 200',
                    default=200,
                    required=False)

parser.add_argument('-s', '--size', type=int,
                    help='Input size of image to be used. [Default] = 200',
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


start_time = time.time()

# input data
path_image = os.path.join(path_data, 'image')
path_label = os.path.join(path_data, 'label')

# Results path
path_result = os.path.join(path_data, 'result')
path_tiled = os.path.join(path_result, 'tiled')
path_predict = os.path.join(path_result, 'prediction')
path_merged_prediction = os.path.join(path_result, 'merged_prediction')

# Tiled path
path_tile_image = os.path.join(path_tiled, 'image/')
path_tile_label = os.path.join(path_tiled, 'label/')

# Output file
file_output = os.path.join(path_merged_prediction, 'output.tif')

# Logging output paths
logger.info('path_image : ' + path_image)
logger.info('path_label : ' + path_label)
logger.info('path_result : ' + path_result)
logger.info('path_tiled : ' + path_tiled)
logger.info('path_predict : ' + path_predict)
logger.info('path_tile_image : ' + path_tile_image)
logger.info('path_tile_label : ' + path_tile_label)
logger.info('Tile image path is %s' % (path_merged_prediction))

# Creating directory
util.check_dir(path_tile_image)
util.check_dir(path_tile_label)
util.check_dir(path_predict)
util.check_dir(path_tiled)
util.check_dir(path_data)
util.check_dir(path_merged_prediction)

# load all the training images
train_set = io.train_data()

# Definging inputs to the class
train_set.path_image = path_tile_image
train_set.path_label = path_tile_label
train_set.image_size = image_size
train_set.max_num_cpu = max_num_cpu
train_set.max_num_gpu = max_num_gpu
accuracy = {}

# Tiling images
logging.info('Tiling Images ...')
logger.info('Tiling Images..')

if skip_gridding == 0:
    tile_image = io.checkres(path_image, image_size,
                             path_tile_image, percent_overlap)
    tile_label = io.checkres(path_label, image_size,
                             path_tile_label, percent_overlap)

logging.info('Tiling Completed')
logger.info('Tiling Completed')

# Logging inputs data
logger.info('Tiling Completed')
logger.info('path_tile_image : ' + str(path_tile_image))
logger.info('path_tile_label : ' + str(path_tile_label))
logger.info('image_size : ' + str(train_set.image_size))
logger.info('max_num_cpu : ' + str(train_set.max_num_cpu))
logger.info('max_num_gpu : ' + str(train_set.max_num_gpu))
logger.info('percent_overlap : ' + str(percent_overlap))

# Listing images
train_set.list_data()
part = len(train_set.image_part_list)
logging.info('Number of parts : %s' % (str(part)))
for k in range(part):

    # Loading the training image and labeled image
    train_image = io.get_image(
        train_set.image_part_list[k], train_set.image_size)
    train_label = io.get_label(
        train_set.label_part_list[k], train_set.image_size)

    shape_train_image = train_image.shape
    shape_train_label = train_label.shape

    # Resizing to correct shape
    train_image = np.resize(train_image, [
                            shape_train_image[0], shape_train_image[1], shape_train_image[2], 3])
    train_label = np.resize(train_label, [
                            shape_train_label[0], shape_train_label[1], shape_train_label[2], 1])

    train_lb = np.zeros(train_label.shape)
    for i in range(shape_train_label[0]):
        temp = train_label[i]
        temp = temp[:, :, 0]
        temp[temp == 255] = 1
        train_lb[i, :, :, 0] = temp

    train_label = train_lb
    # logging.infoing type and number of imgaes and labels
    logging.info("shape of train_image" + str(shape_train_image))
    logging.info("shape of train_label" + str(shape_train_label))
    logger.info("shape of train_image" + str(shape_train_image))
    logger.info("shape of train_label" + str(shape_train_label))

    # get name,size,geo referencing data
    data = io.get_geodata(train_set.image_part_list[k])

    # defining loss functions
    loss_ = loss.dice_coef_loss

    # loading model from model file not weights file
    model = load_model(path_model, custom_objects={
        'dice_coef_loss': loss.dice_coef_loss, 'dice_coef': loss.dice_coef, 'jaccard_coef': loss.jaccard_coef})

    # evaluating model
    eval_score = model.evaluate(
        train_image, train_label, batch_size=16, verbose=1)  # , sample_weight=None, steps=None)

    logging.info('Model loss is : %s' % (eval_score[0]))
    logging.info('Model accuracy is : %s' % (eval_score[1]))
    logging.info('Model dice coefficient is : %s' % (eval_score[2]))
    logging.info('Model Jacard coefficient is : %s' % (eval_score[3]))

    # Logging model accuracies
    logger.info('Model loss is : %s' % (eval_score[0]))
    logger.info('Model accuracy is : %s' % (eval_score[1]))
    logger.info('Model dice coefficient is : %s' % (eval_score[2]))
    logger.info('Model Jacard coefficient is : %s' % (eval_score[3]))

    # Saving accuracies to JSON
    accuracy['Dice coefficient'] = eval_score[2]
    accuracy['Jacard Coefficient'] = eval_score[3]

#     # prediction model
#     predict_result = model.predict(
#         train_image, batch_size=16, verbose=1)  # , steps=None)

#     predict_image = []
#     for i in range(predict_result.shape[0]):
#         # im = train_images[i]
#         lb = predict_result[i, :, :, :]
#         lb = np.round(lb, decimals=0)
#         im_path = os.path.join(path_predict, os.path.basename(data['name'][i]))
#         predict_image.append(im_path)
#         io.write_tif(im_path, lb*255, data['geotransform']
#                      [i], data['geoprojection'][i], data['size'][i])
#     #    cv2.imwrite(im_path,lb*255)

#     # merging = []
#     # output_vrt = os.path.join(path_data, 'merged.vrt')
#     # for root, dirs, files in os.walk(path_predict):
#     #     for file in files:
#     #         if ".tif" in file:
#     #             merging.append(file)

#     # gdal.BuildVRT(output_vrt, merging, options=gdal.BuildVRTOptions(
#     #     srcNodata=-9999, VRTNodata=-9999))

# # Merging all the tif datasets
# logger.info('Merging tiled dataset')
# io.merge_tile(file_output, predict_image)

# # Converting raster to Vector
# logging.info('Converting Raster to vector')
# output_format = 'shp'
# io.raster2vector(file_output, os.path.dirname(file_output), output_format)

# # Post Processing shp to axis aligned bounding box
# postprocess.aabbox(os.path.dirname(file_output), output_format)

# Saving to accuracy.json
io.tojson(accuracy, os.path.join(path_result, 'accuracy.json'))
logger.info('Completed')
sys.exit()
