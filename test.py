from __future__ import print_function
from src import loss, io
#from matplotlib import pyplot as plt

from keras.models import load_model
import os
from keras.models import Model
from keras import backend as K
import cv2
import numpy as np
import time

start_time = time.time()
# training data
image_location = "/home/indshine-2/Music/data2/test/image/"
image = image_location + "tiled"
#label = image_location +"label_tiled"

image_size = 200


class train_data():
    def __init__(self, image, count):
        #    def __init__(self, image, label,count):
        self.image = []
        self.label = []
        self.data = {}

        list_geotransform = []
        list_geoprojection = []
        list_size = []
        list_name = []

        for file in os.listdir(image):
            if file.endswith(".tif"):
                geotransform, geoprojection, size = io.read_tif(image+"/"+file)
                list_geotransform.append(geotransform)
                list_geoprojection.append(geoprojection)
                list_name.append(file)
                list_size.append(size)
#                list_name.append(file)

                self.image.append(cv2.resize(cv2.imread(
                    image+"/"+file), (image_size, image_size)))
                count = count + 1
                if count % 500 == 0:
                    print(count)

        self.data = {'name': list_name,
                     'size': list_size,
                     'geotransform': list_geotransform,
                     'geoprojection': list_geoprojection}

#        for file in os.listdir(label):
#            if file.endswith(".tif"):
#                #print(label+"/"+file)
##                label_array,_,_,_ = read_tif(label + "/" + file)
#                self.label.append(cv2.resize(cv2.imread(label+"/"+file),(image_size,image_size)))

    def get_image(self):
        return np.array(self.image)

#    def get_label(self):
#        return np.array(self.label)

    def get_data(self):
        return self.data


count = 0
# load all the training images
#train_set = train_data(image, label,count)
train_set = train_data(image, count)

# get the training image
train_images = train_set.get_image()
shape = train_images.shape
train_images = np.resize(train_images, [shape[0], shape[1], shape[2], 3])

# get name,size,geo referencing data
data = train_set.get_data()

# get the segmented image
#train_label = train_set.get_label()
#train_label = np.resize(train_label,[shape[0],shape[1],shape[2],1])
loss_ = loss.dice_coef_loss

model_file = '/home/indshine-2/Music/data2/train/result/model/Part2_model.h5'

model = load_model(model_file, custom_objects={
                   'dice_coef_loss': loss.dice_coef_loss, 'dice_coef': loss.dice_coef, 'jaccard_coef': loss.jaccard_coef})

output = model.predict(train_images, batch_size=16, verbose=1)  # , steps=None)
path = '/home/indshine-2/Music/data2/test/image/output/'

if not os.path.exists(path):
    os.makedirs(path)
# loss = model.evaluate(train_images, train_label, batch_size=32, verbose=1)#, sample_weight=None, steps=None)

for i in range(output.shape[0]):
    #    im = train_images[i]
    lb = output[i, :, :, :]
    lb = np.round(lb, decimals=0)
    im_path = path + os.path.basename(data['name'][i])
    io.write_tif(im_path, lb*255, data['geotransform']
                 [i], data['geoprojection'][i], data['size'][i])
#    cv2.imwrite(im_path,lb*255)
