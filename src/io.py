""" Input Output functions"""
import cv2
import gdal
import os
import sys
import numpy as np
from src.gridding import gridding
from src.bf_grid import bf_grid


class train_data():
    def __init__(self):
        self.image_list = []
        self.label_list = []
        self.count = 0

        self.path_image = []
        self.path_label = []
        self.image_size = []
        self.max_num_cpu = 50000.00
        self.max_num_gpu = 6500.00
        self.image_part_list = []
        self.label_part_list = []
        self.geodata = {}

        # storing geo-referencing information


    # Spliting list if it is greater than max_num_cpu
    def split_list(self):

        part = int(math.ceil(len(self.image_list)/self.max_num_cpu))
        length = len(self.image_list)

        return [self.image_list[i*length // part: (i+1)*length // part]
                for i in range(part)]

    # Creating list of data (images and labels)
    def list_data(self):
        list_geotransform = []
        list_geoprojection = []
        list_size = []
        list_name = []
        
        for file in os.listdir(self.path_image):
            if file.endswith(".tif"):
                if os.path.isfile(os.path.abspath(os.path.join(self.path_image, file))) == False:
                    print('File %s not found' %
                          (os.path.abspath(os.path.join(self.path_image, file))))
                    continue
                #    sys.exit('File %s not found'%(os.path.abspath(os.path.join(image_lo,file))))

                if os.path.isfile(os.path.abspath(os.path.join(self.path_label, file))) == False:
                    print('File %s not found' %
                          (os.path.abspath(os.path.join(self.path_label, file))))
                    continue
                #    sys.exit('File %s not found'%(os.path.abspath(os.path.join(label_lo,file))))

                self.image_list.append(os.path.abspath(
                    os.path.join(self.path_label, file)))

                self.label_list.append(os.path.abspath(
                    os.path.join(self.path_label, file)))

                self.count = self.count + 1

                # Storing geo-referencing information
                geotransform, geoprojection, size = read_tif(
                    os.path.join(self.path_image, file))
                list_geotransform.append(geotransform)
                list_geoprojection.append(geoprojection)
                list_name.append(file)
                list_size.append(size)

        # Spliting large number of images into smaller parts to fit in CPU memory
        self.image_part_list = self.split_list()
        self.label_part_list = self.split_list()

        print('Total number of images found: %s' % (self.count))
        print('Total number of splits: %s' % (len(self.image_part_list)))

        self.geodata = {'name': list_name,
                        'size': list_size,
                        'geotransform': list_geotransform,
                        'geoprojection': list_geoprojection}


def get_image(image_list, image_size):  # Loading images from list
    image = []
    print('Reading Images...')
    for i in range(len(image_list)):
        image.append(cv2.resize(cv2.imread(os.path.abspath(
            os.path.join(image_list[i], file))), (image_size, image_size)))

    return np.array(image)


def get_label(label_list, image_size):  # Loading labels from list
    label = []
    print('Reading Labels...')
    for i in range(train_data.count):
        label.append(cv2.resize(cv2.imread(os.path.abspath(
            os.path.join(label_list[i], file)), 0), (image_size, image_size)))

    return np.array(label)


def read_tif(tif_file):
    #    array = cv2.imread(tif_file)
    #    driver = gdal.GetDriverByName("GTiff")
    ds = gdal.Open(tif_file)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    size = arr.shape
    geotransform = ds.GetGeoTransform()
    geoprojection = ds.GetProjection()
    return geotransform, geoprojection, (size[1], size[0])


def write_tif(tif_file, array, geotransform, geoprojection, size):
    array = cv2.resize(array, size)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(tif_file, size[0], size[1], 1)
    outdata.SetGeoTransform(geotransform)  # sets same geotransform as input
    outdata.SetProjection(geoprojection)  # sets same projection as input
    outdata.GetRasterBand(1).WriteArray(array)
#    outdata.GetRasterBand(1).SetNoDataValue(-9999)##if you want these values transparent
    outdata.FlushCache()  # saves to disk!!


def checkdir(path):
    if not os.path.exists(path):
        os.makedirs(os.path.abspath(path))


def checkres(path, size, output, percent_overlap):
    #    inputs = []
    grid_size = size

    # 10% overlap to be taken
    overlap = int(grid_size * percent_overlap)
    bf_grid(path, size, size, overlap, output)
#    for file in os.listdir(path):
#        if file.endswith(".tif"):
#            array = cv2.imread(os.path.join(path,file))
#            shapes = array.shape
#            if max(shapes)>size:
#                inputs.append(os.path.join(path,file))
#
#        elif file.endswith(".tiff"):
#            array = cv2.imread(os.path.join(path,file))
#            shapes = array.shape
#            if max(shapes)>size:
#                inputs.append(os.path.join(path,file))
#        else:
#            return False
#    args = [inputs, grid_size, grid_size, overlap, output]
#    print('Gridding Images ...')
#    gridding(args)
    return True
