from __future__ import logging.info_function
# from matplotlib import pyplot as plt
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
from src import loss, model, io, log, util
from src.bf_grid import bf_grid
import config


util.check_dir(config.path_logs)
util.set_logging(os.path.join(config.path_logs, 'train.log'))

parser = argparse.ArgumentParser(
    description='See description below to see all available options')

parser.add_argument('-m', '--model',
                    help='Continuining training from the given model. \
                          [Default] is no model given',
                    default=float('nan'),
                    required=False)

# Parsing arguments
args = parser.parse_args()
data_lo = args.data
image_size = args.size
num_class = args.classes
skip_gridding = args.skip_gridding
max_num_cpu = args.max_cpu
max_num_gpu = args.max_gpu
grid_size = args.grid_size
percent_overlap = args.overlap
num_epoch = args.epoch
prev_model = args.model


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

# Checking its resolution

""" TO DO Data Generator """
# load all the training images
train_set = io.train_data()

# Definging inputs to the class
train_set.path_image = path_tile_image
train_set.path_label = path_tile_label
train_set.image_size = image_size
train_set.max_num_cpu = max_num_cpu
train_set.max_num_gpu = max_num_gpu
timing = {}

""" Data Generator Ends"""

# Hard Codeded values
# Initializing counting number of images loaded
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

# Listing images
train_set.list_data()

part = len(train_set.image_part_list)
"""
If number of images is greater than max_num_cpu then use loop.
This is done to prevent overflow of RAM.
For 32 GB RAM and 6GB GPU,
max_num_cpu = 50000
max_num_gpu = 6500
"""
for k in range(part):

    # Getting start time of reading images
    st_read_im = time.time()

    # get the training image and segmented image
    train_image = io.get_image(
        train_set.image_part_list[k], train_set.image_size)
    train_label = io.get_label(
        train_set.label_part_list[k], train_set.image_size)

    # Getting total reading images timing
    end_read_im = time.time()
    timing['Read_image_%s' % (str(k))] = end_read_im - st_read_im

    shape_train_image = train_image.shape
    shape_train_label = train_label.shape

    # logging.infoing type and number of imgaes and labels
    logging.info("shape of train_image" + str(shape_train_image))
    logging.info("shape of train_image" + str(shape_train_image))

    logging.info("shape of train_label" + str(shape_train_label))
    logging.info("shape of train_label" + str(shape_train_label))

    # Checking number of classes label has
    if num_class != 1:
        unique = []
        for i in range(shape_train_image[0]):
            unique.append(np.unique(train_label[i, :, :, :]))

        num_class = len(np.unique(np.asarray(unique)))
        logging.info('Num of unique classes identified is %s' % (str(num_class)))
        logging.warning('Multiclass segmentation is not supported yet')

        if num_class > 1:
            logging.info('Multiclass segmentation is not supported yet')
            logging.critical('Multiclass segmentation is not supported yet')
            sys.exit(-1)

    # Checking if number of images and label are same or not
    if shape_train_image[0] != shape_train_label[0]:
        logging.info('Num of images and label doesnt match. Make sure you have same num of image and corresponding labels in the data folder. %s != %s' % (
            str(shape_train_image[0]), str(shape_train_label[0])))
        sys.exit(-1)

    # Spliting number of images if it is greater than 7000 so that it can be fit in GPU memory
    # Maximum array that can be fit in GPU(6GB) 1060
    max_num_gpu = math.ceil(shape_train_image[0]/train_set.max_num_gpu)

    train_image_split = np.array_split(train_image, max_num_gpu)
    train_label_split = np.array_split(train_label, max_num_gpu)

    # logging.infoing type and number of imgaes and labels
    logging.info("shape of Split train_image" + str(train_image_split[0].shape))
    logging.info("shape of Split train_image" +
                 str(train_image_split[0].shape))

    logging.info("shape of Split train_label" + str(train_label_split[0].shape))
    logging.info("shape of Split train_label" +
                 str(train_label_split[0].shape))

    for j in range(max_num_gpu):

        # Loop timings
        st_loop = time.time()

        # Loading previous models
        if j >= 1 or k >= 1:
            """
            Weights are saved in format W_k_j.h5
            where, k is index of larger (cpu loop)
            and j in index of smaller(gpu loop)
            """
            load_weight_file = path_weight
            umodel.load_weights(load_weight_file)

        # Creating temporary train image and train labels.
        temp_train_image = train_image_split[j]
        temp_train_label = train_label_split[j]
        logging.info('temp_train_image : %s' % str(temp_train_image.shape))
        logging.info('temp_train_label : %s' % str(temp_train_label.shape))

        train_im = np.zeros((len(temp_train_image), image_size,
                             image_size, num_image_channels))
        train_lb = np.zeros((len(temp_train_label), image_size,
                             image_size, num_label_channels), dtype=np.bool)

        for i in range(len(temp_train_image)):
            train_im[i, :, :, :] = temp_train_image[i]
            temp = temp_train_label[i]
            temp[temp == 255] = 1
            train_lb[i, :, :, 0] = temp

        # Defining losses and accuracy metrces
        model_loss = loss.dice_coef_loss
        # model_loss = jaccard_loss(100)

        # Only define model
        if j < 1 and k < 1:
            # Defining model
            logging.info('Defining model')
            logging.warning('Defining model')
            umodel = model.unet(image_size)

            # Continue from the previouse defined model
            if prev_model == prev_model:
                umodel.load_weights(prev_model)

        # Compiling model
        umodel.compile(optimizer=Adam(lr=1e-4),  # loss = 'binary_crossentropy', metrics = ['accuracy'])
                       loss=model_loss,
                       metrics=['accuracy', loss.dice_coef, loss.jaccard_coef])

        # create a UNet (512,512)
        # look at the summary of the unet
        umodel.summary()

        # Logging accuracies
        csv_logging = CSVlogging(os.path.join(
            save_csv_lo, 'log_%s_%s.csv' % (str(k), str(j))), separator=',', append=True)

        # -----------errors start here-----------------
        # filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        # fit the unet with the actual image, train_image
        # and the output, train_label
        umodel.fit(train_im,
                   train_lb,
                   batch_size=16,
                   epochs=num_epoch,
                   validation_split=0.15,
                   callbacks=[csv_logging, tensorboard])

        logging.info('Saving model')
        umodel.save(os.path.join(save_model_lo, 'M_%s_%s.h5' %
                                 (str(k), str(j))))

        # Saving path of weigths saved
        logging.info('Saving weights')
        path_weight = os.path.join(
            save_weight_lo,  'W_%s_%s.h5' % (str(k), str(j)))

        umodel.save_weights(path_weight)

        # Counting number of loops
        count = count + 1
        end_loop = time.time()

        # Getting timings
        timing['loop_%s_%s' % (str(k), str(j))] = end_loop - st_loop
        io.tojson(timing, os.path.join(result_lo, 'Timing.json'))

    # Clearing memory
    train_image = []
    train_label = []

end_time = time.time() - st_time
timing['Total Time'] = str(end_time)

# Saving to JSON
io.tojson(timing, os.path.join(result_lo, 'Timing.json'))

# model.evaluate(x=vali_images, y=vali_label, batch_size=32, verbose=1)#, sample_weight=None, steps=None)
# model.predict( vali_images, batch_size=32, verbose=1)#, steps=None)
logging.info('Completed')
sys.exit()
