import os
import argparse
import sys
import logging
import math
import numpy as np
import time

import keras
# Importing Keras
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVlogging, TensorBoard
from keras.models import load_model
from src import loss, model, io, log, util, dataGenerator
from src.bf_grid import bf_grid
import config

timing = {}
util.check_dir(config.path_logs)
util.set_logging(os.path.join(config.path_logs, 'train.log'))

parser = argparse.ArgumentParser(
    description='See description below to see all available options')

parser.add_argument('-p', '--pretrained',
                    help='Continuining training from the given model. \
                          [Default] is no model given',
                    default=None,
                    type=str,
                    required=False)

# Parsing arguments
args = parser.parse_args()

# Checking directories
util.check_dir(config.path_model)
util.check_dir(config.path_weight)
util.check_dir(config.path_prediction)
util.check_dir(config.path_tiled)
util.check_dir(config.path_tile_image)
util.check_dir(config.path_tile_label)


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

# Data Generators
dataList = dataGenerator.getData()
list_IDs, imageMap, labelMap = dataList.getList()

training_generator = dataGenerator.DataGenerator(
    list_IDs=list_IDs, imageMap=imageMap, labelMap=labelMap,
    batch_size=config.batch, n_classes=None,
    image_channels=config.num_image_channels,
    label_channels=config.num_label_channels,
    image_size=None, shuffle=True)

""" Data Generator Ends"""

count = 0
st_time = time.time()

# Logging input data
logging.info('path_tile_image: {}'.format(config.path_tile_image))
logging.info('path_tile_label: {}'.format(config.path_tile_label))
logging.info('image_size: {}'.format(config.image_size))
logging.info('num_image_channels: {}'.format(config.num_image_channels))
logging.info('num_epoch: {}'.format(config.epoch))

# Tensorboard
tensorboard = TensorBoard(
    log_dir=config.path_tensorboard_log, histogram_freq=1)

unet_model = model.unet(config.image_size)

# Listing images
if args.pretrained is not None:
    unet_model.load_weights(args.pretrained)

# Compiling model
unet_model.compile(optimizer=Adam(lr=1e-4),
                   loss='categorical_crossentropy',
                   metrics=[loss.dice_coef, loss.jaccard_coef])

# create a UNet (512,512)
# look at the summary of the unet
unet_model.summary()

# Logging accuracies
csv_logger = keras.callbacks.callbacks.CSVLogger(
    os.path.join(config.path_logs, 'keras_training.log'))

# Creating model callbacks
path_save_callback = os.path.join(
    config.path_weight, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
saving_model = keras.callbacks.callbacks.ModelCheckpoint(path_save_callback,
                                                         monitor='val_loss',
                                                         verbose=0,
                                                         save_best_only=False,
                                                         save_weights_only=True,
                                                         mode='auto',
                                                         period=5)

# fit the unet with the actual image, train_image
# and the output, train_label
history = unet_model.fit_generator(
    generator=training_generator,
    epochs=10,
    workers=3,
    # sample_weight=weight_vector,
    use_multiprocessing=True,
    callbacks=[saving_model, csv_logger]
)

unet_model.fit_generator(generator=training_generator,
                         batch_size=config.batch,
                         epochs=config.epoch,
                         workers=3,

                         validation_split=0.15,
                         callbacks=[csv_logger, tensorboard, saving_model])

# Saving path of weigths saved
logging.info('Saving model')
unet_model.save(os.path.join(config.path_weight, 'final.hdf5'))

# Counting number of loops
count = count + 1
end_loop = time.time()

# Getting timings
end_time = time.time() - st_time
timing['Total Time'] = str(end_time)

# Saving to JSON
io.tojson(timing, os.path.join(config.path_model, 'Timing.json'))
logging.info('Completed')
