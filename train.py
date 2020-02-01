import os
import argparse
import sys
import logging
import math
import numpy as np
import time
import random

# Importing Keras
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from src import metric, model, io, util, dataGenerator, loss
from src.bf_grid import bf_grid
import config


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
_config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


timing = {}
util.check_dir(config.path_logs)
util.set_logger(os.path.join(config.path_logs, 'train.log'))

parser = argparse.ArgumentParser(
    description='See description below to see all available options')

parser.add_argument('-pt', '--pretrained',
                    help='Continuining training from the given model. \
                          [Default] is no model given',
                    default=None,
                    type=str,
                    required=False)

parser.add_argument('-w', '--weight',
                    help='If model provided is Model Weight or not. \
                        True - It is Weight, False- Complete Model',
                    default=None,
                    type=bool,
                    required=False)

# Parsing arguments
args = parser.parse_args()

# Checking directories
util.check_dir(config.path_model)
util.check_dir(config.path_weight)
util.check_dir(config.path_tiled)
util.check_dir(config.path_tiled_image)
util.check_dir(config.path_tiled_label)


# Checking if image or images path exist in data folder
if not os.path.exists(config.path_image):
    msg = '{} does not exist. Ensure that directory exist'.format(
        config.path_image)
    logging.error(msg)
    raise(msg)

# Checking if label or labes path exist in data folder
if not os.path.exists(config.path_label):
    msg = '{} does not exist. Ensure that directory exist'.format(
        config.path_label)
    logging.error(msg)
    raise(msg)

# Writing all parameters into configuration file
configuration = {}


# Training Data Set
training_dataList = dataGenerator.getData(
    path_tile_image=config.path_tiled_image,
    path_tile_label=config.path_tiled_label)

training_list_ids, training_imageMap, training_labelMap = training_dataList.getList()

# Validation Data Set
validation_dataList = dataGenerator.getData(
    path_tile_image=config.path_vali_tiled_image,
    path_tile_label=config.path_vali_tiled_label)

validation_list_ids, validation_imageMap, validation_labelMap = validation_dataList.getList()

# Training DataGenerator
training_generator = dataGenerator.DataGenerator(
    list_IDs=training_list_ids, imageMap=training_imageMap,
    labelMap=training_labelMap,
    batch_size=config.batch, n_classes=None,
    image_channels=config.num_image_channels,
    label_channels=config.num_label_channels,
    image_size=config.image_size, shuffle=True)

# Validation DataGenerator
validation_generator = dataGenerator.DataGenerator(
    list_IDs=validation_list_ids, imageMap=validation_imageMap,
    labelMap=validation_labelMap,
    batch_size=config.batch, n_classes=None,
    image_channels=config.num_image_channels,
    label_channels=config.num_label_channels,
    image_size=config.image_size, shuffle=False)

""" Data Generator Ends"""

st_time = time.time()

# Logging input data
logging.info('path_tiled_image: {}'.format(config.path_tiled_image))
logging.info('path_tiled_label: {}'.format(config.path_tiled_label))
logging.info('image_size: {}'.format(config.image_size))
logging.info('num_image_channels: {}'.format(config.num_image_channels))
logging.info('num_epoch: {}'.format(config.epoch))


unet_model = model.unet(config.image_size)

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

elif args.weight is False:
    try:
        unet_model = load_model(args.pretrained, custom_objects={
            'dice_coef': metric.dice_coef, 'jaccard_coef': metric.jaccard_coef})
    except Exception as e:
        msg = 'Unable to load model: {}'.format(args.pretrained)
        logging.error(msg)
        raise('{}. Error : {}'.format(msg, e))

# Compiling model
unet_model.compile(optimizer=Adam(lr=1e-4),
                   loss=loss.weighted_binary_crossentropy,  # 'binary_crossentropy',  #
                   metrics=[metric.dice_coef, metric.jaccard_coef])

# create a UNet (512,512)
# look at the summary of the unet
unet_model.summary()

# Logging accuracies
csv_logger = keras.callbacks.CSVLogger(
    os.path.join(config.path_logs, 'keras_training.log'))

# Creating model callbacks
path_save_callback = os.path.join(
    config.path_weight, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
saving_model = keras.callbacks.ModelCheckpoint(path_save_callback,
                                               monitor='val_loss',
                                               verbose=0,
                                               save_best_only=False,
                                               save_weights_only=True,
                                               mode='auto',
                                               period=5)

# fit the unet with the actual image, train_image
# and the output, train_label
history = unet_model.fit_generator(generator=training_generator,
                                   epochs=config.epoch,
                                   workers=3,
                                   validation_data=validation_generator,
                                   callbacks=[csv_logger, saving_model])

# Saving path of weigths saved
logging.info('Saving model')
unet_model.save(os.path.join(config.path_weight, 'final.hdf5'))

# Getting timings
end_time = time.time() - st_time
timing['Total Time'] = str(end_time)

# Saving to JSON
io.tojson(timing, os.path.join(config.path_model, 'Timing.json'))
logging.info('Completed')
sys.exit()
