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
import time as mtime
import argparse
# import logging
from src.io import checkdir
from src import postprocess
import gdal
from os.path import basename, normpath, join, splitext, dirname, isfile
# Setup logging
# logger = log.get_logger('testing')
# logger.propagate = False
# log.log2file('Testing')

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
                    help='If gridding is already done then skip it. [Default] is 0 (No)',
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

parser.add_argument('-f', '--format', type=str,
                    help=' Specify the output format of the results. Available options are shp, geojson, kml. [Default] = shp',
                    default='shp',
                    required=False)

parser.add_argument('-lf', '--linearfeature', type=int,
                    help='If data is linear feature. Example in case of road and railways. [Defualt] is 0 (no)',
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
output_format = args.format
linear_feature = args.linearfeature

print('percent_overlap : ' + str(percent_overlap))
st_time = mtime.time()

# Storing time of process here
timing = {}

# Current running process
current_process = []
current_process.append('initilization')

# Filer for post processing
filter = 3
simplify_para = 0.7  # in metres

# input data
path_image = join(path_data, 'image')

# Results path
path_result = join(path_data, 'result')
path_tiled = join(path_result, 'tiled')
path_predict = join(path_result, 'prediction')
path_merged_prediction = join(path_result, 'merged_prediction')

# Tiled path
path_tile_image = join(path_tiled, 'image')

# Output file
file_output = join(path_merged_prediction, 'output.tif')

# Logging output paths
print('Tile image path is %s' % (path_merged_prediction))
print('Tile image path is %s' % (path_tile_image))
print('Predict path is %s' % (path_predict))
print('Result path is %s' % (path_result))
print('Image path is %s' % (path_image))


print('Tiling Images ...')

# Creating directory
checkdir(path_tile_image)
checkdir(path_predict)
checkdir(path_tiled)
checkdir(path_data)
checkdir(path_merged_prediction)

if skip_gridding == 0:
    time = mtime.time()
    current_process.append('tiling')
    tile_image = io.test_checkres(path_image, grid_size,
                                  path_tile_image, percent_overlap)
    timing[current_process[-1]] = mtime.time() - time
print('Tiling Completed')


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
    current_process.append('loading_images')
    train_image = io.get_image(
        train_set.image_part_list[k], train_set.image_size)

    shape_train_image = train_image.shape

    # Printing type and number of imgaes and labels
    print("shape of train_image" + str(shape_train_image))

    train_image = np.resize(train_image, [
                            shape_train_image[0], shape_train_image[1], shape_train_image[2], 3])

    # get name,size,geo referencing data
    data = io.get_geodata(train_set.image_part_list[k])

    # defining loss functions
    loss_ = loss.dice_coef_loss

    # loading model from model file not weights file
    model = load_model(path_model, custom_objects={
        'dice_coef_loss': loss.dice_coef_loss, 'dice_coef': loss.dice_coef, 'jaccard_coef': loss.jaccard_coef})
    current_process.append('loading_model')

    # prediction model
    predict_result = model.predict(
        train_image, batch_size=16, verbose=1)  # , steps=None)
    current_process.append('predicting')

    print('Saving Prediction...')
    predict_image = []
    for i in range(predict_result.shape[0]):
        # im = train_images[i]
        lb = predict_result[i, :, :, :]
        lb = np.round(lb, decimals=0)
        path_im = join(
            path_predict, basename(normpath(dirname(train_set.image_part_list[k][i]))), basename(data['name'][i]))
        checkdir(os.path.dirname(path_im))
        predict_image.append(path_im)

        # Saving data to disk
        current_process.append('saving_prediction')
        io.write_tif(path_im, lb*255, data['geotransform']
                     [i], data['geoprojection'][i], data['size'][i])
        current_process.append('saving_prediction')
    
    # Flushing all the memory 
    train_image = []
    predict_result = []
    lb = []

timing['Processing'] = mtime.time() - st_time


# Merging tiled dataset to single tif
time = mtime.time()
print('Merging and compressing %s tiled dataset. This may take a while' % (
    str(train_set.count)))
current_process.append('merging')

path_merged = []
for root, dirs, files in os.walk(path_predict):
    _data = []
    temp = join(path_merged_prediction, basename(normpath(root)) + '.tif')
    path_merged.append(temp)

    for file in files:
        if file.endswith(".tif") or file.endswith(".tiff"):
            _data.append(join(root, file))
    try:
        io.merge_tile(temp, _data)
    except:
        print('Warming! No data found in %s' % (root))
        continue
# merging completed
timing[current_process[-1]] = mtime.time() - time

# Post Processing output image
if linear_feature == 0:

    # Post processing erosion
    print('Post Processing erosion')
    current_process.append('erosion')
    time = mtime.time()

    path_erosion = join(path_merged_prediction, 'erosion')
    checkdir(path_erosion)

    file_erosion = []
    for j in range(len(path_merged)):
        if isfile(path_merged[j]) is True:
            temp = join(
                path_erosion, basename(path_merged[j]))
            file_erosion.append(temp)
            file_output = postprocess.erosion(path_merged[j], filter, temp)
        else:
            print('Warning! file %s not found' % (path_merged[j]))
            continue

    # Erosion completed
    timing[current_process[-1]] = mtime.time() - time

    # Watershed segmentation
    neighbour = 4
    print('Post Processing watershed_segmentation')
    current_process.append('watershed_segmentation')
    time = mtime.time()

    path_watershed = join(path_merged_prediction, 'watershed')
    checkdir(path_watershed)
    file_watershed = []

    for j in range(len(file_erosion)):
        temp = join(
            path_watershed, basename(file_erosion[j]))
        file_watershed.append(temp)
        file_output = postprocess.waterseg(
            file_erosion[j], neighbour, temp)

    # Watershed segmentation completed
    timing[current_process[-1]] = mtime.time() - time

    # Converting raster to Vector
    time = mtime.time()
    print('Converting Raster to vector')
    current_process.append('vectorization')

    path_vector = join(path_merged_prediction, 'vector')
    checkdir(path_vector)
    file_vector = []

    for j in range(len(file_watershed)):
        path_r2v = io.raster2vector(
            file_watershed[j], path_vector, output_format)
        for i in range(len(path_r2v)):
            file_vector.append(path_r2v[i])

    # Vectorization completed
    timing[current_process[-1]] = mtime.time() - time

    path_simplify = join(path_merged_prediction, 'simplify')
    checkdir(path_simplify)
    file_simplify = []

    # Simplification of polygons
    for j in range(len(file_vector)):
        temp = join(path_simplify, basename(file_vector[j]))
        print(temp)
        print(file_vector[j])
        file_simplify.append(temp)
        postprocess.simplify_polygon(file_vector[j], simplify_para, temp)

    # Shp to axis aligned bounding box
    print('Post Processing bounding box')
    current_process.append('aabbox')
    time = mtime.time()

    path_bbox = join(path_merged_prediction, 'bbox')
    checkdir(path_bbox)
    file_bbox = []
    for j in range(len(file_simplify)):
        temp = join(path_bbox, basename(file_simplify[j]))
        print(temp)
        print(file_simplify[j])
        file_bbox.append(temp)
        postprocess.aabbox(file_simplify[j], temp)

    # aabbox completed
    timing[current_process[-1]] = mtime.time() - time


elif linear_feature == 1:
    print('Post Processing skeletonization')
    current_process.append('skeletonization')
    time = mtime.time()

    path_skeleton = join(path_merged_prediction, 'path_skeleton')
    checkdir(path_skeleton)

    file_skeleton = []
    for j in range(len(path_merged)):
        temp = join(
            path_skeleton, basename(path_merged[j]))
        file_skeleton.append(temp)
        _ = postprocess.skeletonize(path_merged[j], temp)

    # Skeletonization completed
    timing[current_process[-1]] = mtime.time() - time

    # Converting raster to Vector
    time = mtime.time()

    print('Converting Raster to vector')
    path_vector = join(path_merged_prediction, 'vector')
    checkdir(path_vector)
    file_vector = []

    for j in range(len(file_skeleton)):
        temp = file_skeleton[j]
        path_r2v = io.raster2vector(
            temp, path_vector, output_format)

    # Vectorization completed
    timing[current_process[-1]] = mtime.time() - time


# Saving to JSON
io.tojson(timing, join(path_result, 'Timing.json'))

print('Process Completed')
sys.exit()
