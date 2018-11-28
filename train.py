from __future__ import print_function
# from matplotlib import pyplot as plt
import os
import argparse
import sys
import logging
import math
import numpy as np

# Importing Keras
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from src.io import checkdir
from src import loss, model, io, log
from src.bf_grid import bf_grid

logger = log.get_logger('training')
log.log2file('training')

parser = argparse.ArgumentParser(
    description='See description below to see all available options')

parser.add_argument('-d', '--data',
                    help='Input directory containing image and label',
                    required=True)

parser.add_argument('-s', '--size', type=int,
                    help='Input size of image to be used. [Default] = 200',
                    default=200,
                    required=False)

parser.add_argument('-c', '--classes', type=int,
                    help='Input number of classes.[Default] = 1',
                    default=1,
                    required=False)

parser.add_argument('-sg', '--skip_gridding', type=int,
                    help='If gridding is already done then skip it. [Default] is No = 0',
                    default=0,
                    required=False)


# Parsing arguments
args = parser.parse_args()
data_lo = args.data
image_size = args.size
num_class = args.classes
skip_gridding = args.skip_gridding

# Output locations
result_lo = os.path.join(data_lo, 'result')
save_model_lo = os.path.join(result_lo, 'model')
save_csv_lo = os.path.join(result_lo, 'accuracy')
save_weight_lo = os.path.join(result_lo, 'weight')
save_prediction_lo = os.path.join(result_lo, 'predicted')
save_tile_lo = os.path.join(result_lo, 'tiled')

# Checking directories
checkdir(result_lo)
checkdir(save_model_lo)
checkdir(save_csv_lo)
checkdir(save_weight_lo)
checkdir(save_prediction_lo)
checkdir(save_tile_lo)

# Hard Codeded values
# Initializing counting number of images loaded
count = 0
max_num_cpu = 50000
max_num_gpu = 6500.00
num_image_channels = 3
num_label_channels = 1
num_epoch = 100
percent_overlap = 0.30

# Defining input paths
image_lo = os.path.join(data_lo, 'image')
label_lo = os.path.join(data_lo, 'label')

# Checking if image or images path exist in data folder
if not os.path.exists(image_lo):
    image_lo = os.path.join(data_lo, 'images')
    if not os.path.exists(image_lo):
        print('image/s path inside %s location doesnt exist. Make sure you have folder name image or images inside %s' % (data_lo, data_lo))
        sys.exit()

# Checking if label or labes path exist in data folder
if not os.path.exists(label_lo):
    label_lo = os.path.join(data_lo, 'labels')
    if not os.path.exists(label_lo):
        print('label/s path inside %s location doesnt exist. Make sure you have folder name label or labels inside %s' % (data_lo, data_lo))
        sys.exit()

# Checking its resolution
path_tile_image = os.path.join(save_tile_lo, 'image/')
path_tile_label = os.path.join(save_tile_lo, 'label/')

checkdir(path_tile_image)
checkdir(path_tile_label)

print('Tiling Images ...')

if skip_gridding == 0:
    tile_image = io.checkres(image_lo, image_size,
                             path_tile_image, percent_overlap)
    tile_label = io.checkres(label_lo, image_size,
                             path_tile_label, percent_overlap)

print('Tiling Completed')
# if tile_image == True:
#    image_lo = path_tile_image
#    label_lo = path_tile_label


# load all the training images
train_set = io.train_data()

# Definging inputs to the class
train_set.path_image = path_tile_image
train_set.path_label = path_tile_label
train_set.image_size = 200
train_set.max_num_cpu = 50000.00
train_set.max_num_gpu = 6500.00

# Listing images
train_set.list_data()

part = len(train_set.image_part_list)
print(part)
"""
If number of images is greater than max_num_cpu then use loop.
This is done to prevent overflow of RAM.
For 32 GB RAM and 6GB GPU,
max_num_cpu = 50000
max_num_gpu = 6500
"""
for k in range(part):

    # get the training image and segmented image
    train_image = io.get_image(
        train_set.image_part_list[k], train_set.image_size)
    train_label = io.get_label(
        train_set.label_part_list[k], train_set.image_size)

    shape_train_image = train_image.shape
    shape_train_label = train_label.shape

    # Printing type and number of imgaes and labels
    print("shape of train_image" + str(shape_train_image))
    print("shape of train_label" + str(shape_train_label))

    # Checking number of classes label has
    if num_class != 1:
        unique = []
        for i in range(shape_train_image[0]):
            unique.append(np.unique(train_label[i, :, :, :]))

        num_class = len(np.unique(np.asarray(unique)))
        print('Num of unique classes identified is %s' % (str(num_class)))
        if num_class > 1:
            print('Multiclass segmentation is not supported yet')
            sys.exit()

    # Checking if number of images and label are same or not
    if shape_train_image[0] != shape_train_label[0]:
        print('Num of images and label doesnt match. Make sure you have same num of image and corresponding labels in the data folder. %s != %s' % (
            str(shape_train_image[0]), str(shape_train_label[0])))
        sys.exit()

    # Spliting number of images if it is greater than 7000 so that it can be fit in GPU memory
    # Maximum array that can be fit in GPU(6GB) 1060
    max_num_gpu = math.ceil(shape_train_image[0]/train_set.max_num_gpu)

    train_image_split = np.array_split(train_image, train_set.max_num_gpu)
    train_label_split = np.array_split(train_label, train_set.max_num_gpu)

    # Printing type and number of imgaes and labels
    print("shape of Split train_image" + str(train_image_split[0].shape))
    print("shape of Split train_label" + str(train_label_split[0].shape))

    for j in range(max_num_gpu):

        # Loading previous models
        if j >= 1:
            """
            Weights are saved in format W_k_j.h5
            where, k is index of larger (cpu loop)
            and j in index of smaller(gpu loop)
            """
            load_weight_file = os.path.join(
                save_weight_lo, 'W_%s_%s.h5' % (str(k+1) + str(j-1)))
            umodel.load_weights(load_weight_file)

        # Creating temporary train image and train labels.
        temp_train_image = train_image_split[j]
        temp_train_label = train_label_split[j]
        print(temp_train_label.shape, temp_train_image.shape)

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
            umodel = model.unet(image_size)

        # Compiling model
        umodel.compile(optimizer=Adam(lr=1e-4),  # loss = 'binary_crossentropy', metrics = ['accuracy'])
                       loss=model_loss,
                       metrics=['accuracy', loss.dice_coef, loss.jaccard_coef])

        # create a UNet (512,512)
        # look at the summary of the unet
        umodel.summary()

        # Logging accuracies
        csv_logger = CSVLogger(os.path.join(
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
                   callbacks=[csv_logger])

        umodel.save(os.path.join(save_model_lo, 'M_%s_%s.h5' %
                                 (str(k), str(j))))
        umodel.save_weights(os.path.join(
            save_weight_lo,  'W_%s_%s.h5' % (str(k+1) + str(j-1))))

# model.evaluate(x=vali_images, y=vali_label, batch_size=32, verbose=1)#, sample_weight=None, steps=None)
# model.predict( vali_images, batch_size=32, verbose=1)#, steps=None)
