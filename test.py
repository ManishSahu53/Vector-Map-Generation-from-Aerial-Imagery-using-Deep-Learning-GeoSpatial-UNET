import os
import sys
import gdal
import cv2
import numpy as np
import time as mtime
import argparse
import logging

# from matplotlib import pyplot as plt
from keras.models import load_model
from keras.models import Model
from keras import backend as K
import tensorflow as tf

from src import postprocess
from src import metric
from src import io
from src import util
from src import bf_grid
from src import metric
from src import dataGenerator
from src import model

import config

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
_config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


util.check_dir(config.path_logs)
util.set_logger(os.path.os.path.join(config.path_logs, 'testing.log'))

parser = argparse.ArgumentParser(
    description='See description below to see all available options')

parser.add_argument('-sg', '--skipGridding',
                    help='If skipping grididing while testing. [Default] False',
                    type=bool,
                    default=False,
                    required=False)

parser.add_argument('-d', '--data',
                    help='Input Data folder where TIF files are stored',
                    type=str,
                    required=True)

parser.add_argument('-pt', '--pretrained',
                    help='Path of pretrained complete model or weight file.\
                         Use -w flag to mark it as weight or complete model',
                    type=str,
                    required=True)

parser.add_argument('-w', '--weight',
                    help='If model provided is Model Weight or not. \
                        True - It is Weight, False- Complete Model',
                    type=bool,
                    required=True)

parser.add_argument('-lf', '--linearFeature',
                    help='If the object is linear feature like road? \
                        [Default] False',
                    type=bool,
                    default=False,
                    required=False)

parser.add_argument('-o', '--output',
                    help='Output Data folder where TIF files will be saved',
                    type=str,
                    required=True)


args = parser.parse_args()
path_data = args.data
st_time = mtime.time()

logging.info('Input data given: {}'.format(path_data))
logging.info('percent_overlap : {}'.format(config.overlap))

# Storing time of process here
timing = {}

# Current running process
logging.info('Initilization')

# Filer for post processing
filter = config.erosion_filter
simplify_parameter = config.simplify_parameter  # in metres

# Results path
path_result = args.output
path_tiled = os.path.join(path_result, 'tiled')
path_predict = os.path.join(path_result, 'prediction')
path_merged_prediction = os.path.join(path_result, 'merged_prediction')
path_erosion = os.path.join(path_merged_prediction, 'erosion')
path_watershed = os.path.join(path_merged_prediction, 'watershed')
path_vector = os.path.join(path_merged_prediction, 'vector')
path_simplify = os.path.join(path_merged_prediction, 'simplify')
path_bbox = os.path.join(path_merged_prediction, 'bbox')

# Creating directory
util.check_dir(path_result)
util.check_dir(path_predict)
util.check_dir(path_tiled)
util.check_dir(path_merged_prediction)
util.check_dir(path_erosion)
util.check_dir(path_watershed)
util.check_dir(path_vector)
util.check_dir(path_simplify)
util.check_dir(path_bbox)

# Logging output paths
logging.info('Result path is %s' % (path_result))
logging.info('Predict path is %s' % (path_predict))
logging.info('Tile image path is %s' % (path_merged_prediction))
logging.info('Erosion path is %s' % (path_erosion))
logging.info('watershed path is %s' % (path_watershed))
logging.info('vector path is %s' % (path_vector))
logging.info('simplify path is %s' % (path_simplify))
logging.info('bbox path is %s' % (path_bbox))


# loading model from model file or  weights file
logging.info('Loading trained model')

if args.weight is True:
    unet_model = model.unet(config.image_size)
    try:
        unet_model.load_weights(args.pretrained)
    except Exception as e:
        msg = 'Unable to load model weights: {}'.format(args.pretrained)
        logging.error(msg)
        raise('{}. Error : {}'.format(msg, e))

else:
    try:
        unet_model = load_model(args.pretrained, custom_objects={
            'dice_coef': metric.dice_coef, 'jaccard_coef': metric.jaccard_coef})
    except Exception as e:
        msg = 'Unable to load model: {}'.format(args.pretrained)
        logging.error(msg)
        raise('{}. Error : {}'.format(msg, e))


# Iterating over all the files
for root, dirs, files in os.walk(path_data):
    for file in files:
        if file.endswith(config.image_ext1) or \
                file.endswith(config.image_ext2):

            temp_path_data = os.path.join(root, file)

            # Creating a new folder for tiled and prediction of the file
            _name = os.path.splitext(os.path.basename(file))[0]
            temp_path_tiled = os.path.join(path_tiled, os.path.basename(_name))
            temp_path_predict = os.path.join(
                path_predict, os.path.basename(_name))

            # Creating a folder in the name of file
            util.check_dir(temp_path_tiled)
            util.check_dir(temp_path_predict)

            logging.info('Gridding image : {}'.format(temp_path_data))
            if args.skipGridding is False:
                time = mtime.time()

                bf_grid.grid_file(path_data=temp_path_data,
                                  path_output=temp_path_tiled)

            logging.info('Gridding Completed: {}'.format(temp_path_data))

            # Loading the Testing image
            logging.info('Reading Gridded image: {}'.format(temp_path_data))

            testing_dataList = dataGenerator.getTestingData(
                path_tile_image=temp_path_tiled)

            testing_list_ids, testing_imageMap = testing_dataList.getList()
            logging.info('Total number of files gridded for {} : {}'.format(
                temp_path_data, len(testing_list_ids)))

            # get name, size, geo referencing data map
            testing_geoMap = io.getGeodata(testing_imageMap)

            # Testing DataGenerator
            training_generator = dataGenerator.DataGenerator(
                list_IDs=testing_list_ids,
                imageMap=testing_imageMap,
                labelMap=None,
                batch_size=config.batch,
                n_classes=None,
                image_channels=config.num_image_channels,
                label_channels=None,
                image_size=config.image_size,
                prediction=True,
                shuffle=False)

            # Prediction model
            logging.info('Predicting data: {}'.format(temp_path_data))
            predictResult = unet_model.predict_generator(
                generator=training_generator,
                workers=6,
                use_multiprocessing=True,
                verbose=1
            )
            logging.info('Number of data files predicted for {} : {} is shape of predicted matrix'.format(
                temp_path_data, predictResult.shape))
            logging.info('Saving Prediction: {}'.format(temp_path_data))
            predict_image = []

            # Iterating over predictions and saving it to geoReferenced TIF files
            temp_listPrediction = []
            for i in range(len(testing_list_ids)):
                file_name = os.path.basename(testing_geoMap[i]['path'])

                labelPrediction = predictResult[i, :, :, :]
                # Setting 0.5 as threshold
                labelPrediction = np.round(labelPrediction, decimals=0)

                temp_path_output = os.path.join(temp_path_predict, file_name)

                # Saving data to disk
                io.write_tif(temp_path_output, labelPrediction*255, testing_geoMap[i]['geoTransform'],
                             testing_geoMap[i]['geoProjection'], testing_geoMap[i]['size'])

                temp_listPrediction.append(temp_path_output)

            timing['Processing'] = mtime.time() - st_time

            # Merging Gridded dataset to single TIF
            time = mtime.time()
            logging.info('Merging and compressing gridded dataset: {}. \
                Total number of files: {}. This may take a while'.format(temp_path_data, len(temp_listPrediction)))

            temp_merged_output = os.path.join(path_merged_prediction, file)
            io.mergeTile(listTIF=temp_listPrediction,
                         path_output=temp_merged_output)

            # # merging completed
            # timing[current_process[-1]] = mtime.time() - time

            temp_erosion_output = os.path.join(path_erosion, file)

            # Post Processing output image
            if args.linearFeature is False:

                # Post processing erosion
                logging.info('Post Processing erosion')
                time = mtime.time()

                postprocess.erosion(path_input=temp_merged_output,
                                    filter=filter,
                                    path_output=temp_erosion_output
                                    )

                # Erosion completed
                logging.info(
                    'Erosion has been completed: {}'.format(temp_path_data))

                # Watershed segmentation
                neighbour = config.watershed_neighbour
                logging.info('Post Processing watershed_segmentation')
                time = mtime.time()

                temp_watershed_output = os.path.join(path_watershed, file)

                # Processing Watershed Segmentation
                postprocess.watershedSegmentation(
                    temp_erosion_output, neighbour, temp_watershed_output)

                # Watershed segmentation completed
                logging.info(
                    'Watershed Segmentation Completed: {}'.format(temp_path_data))

                timing['WatershedSegmentation'] = mtime.time() - time

                # Converting raster to Vector
                time = mtime.time()
                logging.info('Converting Raster to vector')

                temp_raster2vector_output = os.path.join(path_vector, _name)

                print('Input to raster2vector: {}'.format(temp_watershed_output))

                temp_raster2vector_output = io.raster2vector(path_raster=temp_watershed_output,
                                                             path_output=temp_raster2vector_output)
                logging.info(
                    'raster2vector Completed: {}'.format(temp_path_data))

                # # Vectorization completed
                # timing[current_process[-1]] = mtime.time() - time

                # Since we are making a vector file for each raster band, thus there can be multiple... \
                # ... vector files for one raster

                # Simplification of polygons
                logging.info('Simplifying Vectors')
                for _temp_vector_path in temp_raster2vector_output:
                    temp_simplify_output = os.path.join(
                        path_simplify, _name, os.path.basename(_temp_vector_path))

                    util.check_dir(os.path.dirname(temp_simplify_output))

                    postprocess.simplify_polygon(path_shp=_temp_vector_path,
                                                 parameter=config.simplify_parameter,
                                                 path_output=temp_simplify_output)

                    logging.info(
                        'Vector simplification Completed: {}'.format(temp_path_data))

                    # Shp to axis aligned bounding box
                    logging.info(
                        'Post Processing vectors to bounding box')

                    # current_process.append('aabbox')
                    time = mtime.time()

                    temp_bbox_output = os.path.join(
                        path_bbox, _name, os.path.basename(_temp_vector_path))

                    util.check_dir(os.path.dirname(temp_bbox_output))

                    postprocess.aabbox(path_shp=temp_simplify_output,
                                       path_output=temp_bbox_output)

                    logging.info(
                        'AA BBox Completed: {}'.format(temp_path_data))

                # aabbox completed
                # timing[current_process[-1]] = mtime.time() - time

            elif args.linearFeature is True:
                logging.info('Post Processing skeletonization')
                time = mtime.time()

                path_skeleton = os.path.join(
                    path_merged_prediction, 'path_skeleton')
                util.check_dir(path_skeleton)

                temp_skeletonize_output = os.path.join(path_skeleton, file)
                postprocess.skeletonize(
                    path_input=temp_erosion_output, path_output=temp_skeletonize_output)

                # Skeletonization completed
                # timing[current_process[-1]] = mtime.time() - time

                # Converting raster to Vector
                time = mtime.time()

                logging.info('Converting Raster to vector')
                temp_raster2vector_output = os.path.join(path_vector, _name)

                io.raster2vector(path_raster=temp_skeletonize_output,
                                 path_output=temp_raster2vector_output)
                logging.info(
                    'raster2vector Completed: {}'.format(temp_path_data))

                # Vectorization completed
                # timing[current_process[-1]] = mtime.time() - time

            # Saving to JSON
            io.tojson(timing, os.path.join(path_result, 'Timing.json'))

            logging.info('Process Completed')

sys.exit()
