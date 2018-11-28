""" Input Output functions"""
import cv2
import gdal
import os
import sys
import numpy as np
from src.gridding import gridding
from src.bf_grid import bf_grid


class train_data():
    def __init__(self, image_lo, label_lo, image_size, count):
        self.image = []
        self.label = []
        for file in os.listdir(image_lo):
            if file.endswith(".tif"):
                if os.path.isfile(os.path.abspath(os.path.join(image_lo, file))) == False:
                    print('File %s not found' %
                          (os.path.abspath(os.path.join(image_lo, file))))
                    continue
#                    sys.exit('File %s not found'%(os.path.abspath(os.path.join(image_lo,file))))

                if os.path.isfile(os.path.abspath(os.path.join(label_lo, file))) == False:
                    print('File %s not found' %
                          (os.path.abspath(os.path.join(label_lo, file))))
                    continue
#                    sys.exit('File %s not found'%(os.path.abspath(os.path.join(label_lo,file))))

                self.image.append(cv2.resize(cv2.imread(os.path.abspath(
                    os.path.join(image_lo, file))), (image_size, image_size)))
                self.label.append(cv2.resize(cv2.imread(os.path.abspath(
                    os.path.join(label_lo, file)), 0), (image_size, image_size)))
                count = count + 1
                if count > 50000:
                    break
                # Printing num of images loaded
                if count % 500 == 0:
                    print(count)

    def get_image(self):
        return np.array(self.image)

    def get_label(self):
        return np.array(self.label)


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
