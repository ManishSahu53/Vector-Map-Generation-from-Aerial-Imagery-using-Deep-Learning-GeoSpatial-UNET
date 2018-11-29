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
logger = log.get_logger('evaluating')
log.log2file('evaluating')

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

# Parsing arguments
args = parser.parse_args()
path_data = args.data
image_size = args.size
skip_gridding = args.skip_gridding
path_model = args.model


start_time = time.time()

# input data
path_image = os.path.join(path_data, 'image')
path_label = os.path.join(path_data, 'label')

# Results path
path_result = os.path.join(path_data, 'result')
path_tiled = os.path.join(path_result, 'tiled')
path_predict = os.path.join(path_result, 'prediction')

# Tiled path
path_tile_image = os.path.join(path_tiled, 'image/')
path_tile_label = os.path.join(path_tiled, 'label/')

# Logging output paths
logger.info('path_image : ' + path_image)
logger.info('path_label : ' + path_label)
logger.info('path_result : ' + path_result)
logger.info('path_tiled : ' + path_tiled)
logger.info('path_predict : ' + path_predict)
logger.info('path_tile_image : ' + path_tile_image)
logger.info('path_tile_label : ' + path_tile_label)

# Creating directory
checkdir(path_tile_image)
checkdir(path_tile_label)
checkdir(path_predict)
checkdir(path_tiled)
checkdir(path_data)

# load all the training images
train_set = io.train_data()

# Definging inputs to the class
train_set.path_image = path_tile_image
train_set.path_label = path_tile_label
train_set.image_size = 200
train_set.max_num_cpu = 50000.00
train_set.max_num_gpu = 6500.00
accuracy = {}
percent_overlap = 0.0

# Tiling images
print('Tiling Images ...')
logger.info('Tiling Images..')

if skip_gridding == 0:
    tile_image = io.checkres(path_image, image_size,
                             path_tile_image, percent_overlap)
    tile_label = io.checkres(path_label, image_size,
                             path_tile_label, percent_overlap)

print('Tiling Completed')
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
print('Number of parts : %s' % (str(part)))
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
    label_image = np.resize(train_label, [
                            shape_train_label[0], shape_train_label[1], shape_train_label[2], 1])

    # Printing type and number of imgaes and labels
    print("shape of train_image" + str(shape_train_image))
    print("shape of train_label" + str(shape_train_label))
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
        train_image, label_image, batch_size=16, verbose=1)  # , sample_weight=None, steps=None)

    print('Model loss is : %s' % (eval_score[0]))
    print('Model accuracy is : %s' % (eval_score[1]))
    print('Model dice coefficient is : %s' % (eval_score[2]))
    print('Model Jacard coefficient is : %s' % (eval_score[3]))

    # Logging model accuracies
    logger.info('Model loss is : %s' % (eval_score[0]))
    logger.info('Model accuracy is : %s' % (eval_score[1]))
    logger.info('Model dice coefficient is : %s' % (eval_score[2]))
    logger.info('Model Jacard coefficient is : %s' % (eval_score[3]))

    # Saving accuracies to JSON
    accuracy['Dice coefficient'] = eval_score[2]
    accuracy['Jacard Coefficient'] = eval_score[3]

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
    #    cv2.imwrite(im_path,lb*255)

    # merging = []
    # output_vrt = os.path.join(path_data, 'merged.vrt')
    # for root, dirs, files in os.walk(path_predict):
    #     for file in files:
    #         if ".tif" in file:
    #             merging.append(file)

    # gdal.BuildVRT(output_vrt, merging, options=gdal.BuildVRTOptions(
    #     srcNodata=-9999, VRTNodata=-9999))

io.tojson(accuracy, os.path.join(path_result, 'accuracy.json'))
