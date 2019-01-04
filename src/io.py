""" Input Output functions"""
import cv2
import gdal
import os
import sys
import numpy as np
from src.gridding import gridding
from src.bf_grid import bf_grid
from src.bf_grid import test_grid
import math
import json
import ogr
import osr


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

    # Spliting list if it is greater than max_num_cpu
    def split_list(self, list):

        length = len(list)
        part = int(math.ceil(length/self.max_num_cpu))

        return [list[i*length // part: (i+1)*length // part] for i in range(part)]

    # Creating list of data (images and labels)
    def list_data(self):
        for root_image, dirs, files in os.walk(self.path_image):
            for file in files:
                if file.endswith(".tif") or file.endswith(".tiff"):
                    if os.path.isfile(os.path.abspath(os.path.join(root_image, file))) == False:
                        print('File %s not found' %
                              (os.path.abspath(os.path.join(root_image, file))))
                        continue
                    #    sys.exit('File %s not found'%(os.path.abspath(os.path.join(image_lo,file))))

                    """
                    Only load labels with path is given. Else skip. 
                    This is because in case of testing, we dont have test labels
                    """
                    if len(self.path_label) != 0:
                        if os.path.isfile(os.path.abspath(os.path.join(os.path.join(self.path_label, os.path.basename(root_image)), file))) == False:
                            print('File %s not found' %
                                  (os.path.abspath(os.path.join(os.path.join(self.path_label, os.path.basename(root_image)), file))))
                            continue
                        #    sys.exit('File %s not found'%(os.path.abspath(os.path.join(label_lo,file))))
                    else:
                        continue

                    self.image_list.append(os.path.abspath(
                        os.path.join(root_image, file)))

                    self.label_list.append(os.path.abspath(
                        os.path.join(os.path.join(self.path_label, os.path.basename(root_image)), file)))

                    self.count = self.count + 1

        # Spliting large number of images into smaller parts to fit in CPU memory
        self.image_part_list = self.split_list(self.image_list)
        self.label_part_list = self.split_list(self.label_list)
        if self.count == 0:
            sys.exit('No Images found')
        print('Total number of images found: %s' % (self.count))
        print('Total number of splits: %s' % (len(self.image_part_list)))


# Loading images from list
def get_image(image_list, image_size):
    image = []
    print('Reading Images...')
    for i in range(len(image_list)):
        if i % 500 == 0:
            print('Reading %s, %s' % (str(i), os.path.basename(image_list[i])))
        image.append(cv2.resize(cv2.imread(os.path.abspath(
            image_list[i])), (image_size, image_size)))

    return np.array(image)


# Loading labels from list
def get_label(label_list, image_size):
    label = []
    print('Reading Labels...')
    for i in range(len(label_list)):
        if i % 500 == 0:
            print('Reading %s, %s' % (str(i), os.path.basename(label_list[i])))
        gt, gp, s, lb = read_tif(os.path.abspath(label_list[i]))
        label.append(cv2.resize(lb, (image_size, image_size)))

    return np.array(label)


# storing geo-referencing information
def get_geodata(image_list):
    geotransform_list = []
    geoprojection_list = []
    size_list = []
    name_list = []
    geodata = {}

    # Storing geo-referencing information
    for i in range(len(image_list)):
        geotransform, geoprojection, size, _ = read_tif(image_list[i])
        geotransform_list.append(geotransform)
        geoprojection_list.append(geoprojection)
        name_list.append(os.path.basename(image_list[i]))
        size_list.append(size)

    geodata = {'name': name_list,
               'size': size_list,
               'geotransform': geotransform_list,
               'geoprojection': geoprojection_list}

    return geodata


# Reading raster dataset
def read_tif(path_tif):
    #    array = cv2.imread(tif_file)
    #    driver = gdal.GetDriverByName("GTiff")
    ds = gdal.Open(path_tif)
    num_band = ds.RasterCount
    col = ds.RasterXSize
    row = ds.RasterYSize
    array = np.zeros([row, col, num_band])
    for i in range(num_band):
        band = ds.GetRasterBand(i+1)
        arr = band.ReadAsArray()
        array[:, :, i] = arr
    size = arr.shape
    geotransform = ds.GetGeoTransform()
    geoprojection = ds.GetProjection()
    return geotransform, geoprojection, (size[1], size[0]), array


# Writing raster dataset
def write_tif(path_tif, array, geotransform, geoprojection, size):
    dim_array = array.shape
    if len(dim_array) > 2:
        depth = dim_array[2]
    else:
        depth = 1

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(
        path_tif, size[0], size[1], depth, gdal.GDT_Float32)

    # sets same geotransform as input
    outdata.SetGeoTransform(geotransform)
    outdata.SetProjection(geoprojection)  # sets same projection as input
    for i in range(depth):
        try:
            arr = array[:, :, i]
        except:
            arr = array[:, :]
        arr = cv2.resize(arr, size)
        outdata.GetRasterBand(i+1).WriteArray(arr)
    # outdata.GetRasterBand(1).SetNoDataValue(-9999)##if you want these values transparent
    outdata.FlushCache()  # saves to disk!!


# Checking and creating directory in the path
def checkdir(path):
    if not os.path.exists(path):
        os.makedirs(os.path.abspath(path))


# Checking resolution of the input image
def checkres(path, size, output, percent_overlap):
    #    inputs = []
    grid_size = size

    # 10% overlap to be taken
    overlap = int(grid_size * percent_overlap/100)
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


def test_checkres(path, size, output, percent_overlap):
    #    inputs = []
    grid_size = size

    # 10% overlap to be taken
    overlap = int(grid_size * percent_overlap/100)
    test_grid(path, size, size, overlap, output)
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


# Saving to JSON
def tojson(dictA, file_json):
    with open(file_json, 'w') as f:
        json.dump(dictA, f, indent=4, separators=(',', ': '),
                  ensure_ascii=False)


# Merging all the tiled tif to single tif using gdalbuildvrt and gdaltranslate
def merge_tile(path_output, list_tif):
    # Building VRT
    temp = gdal.BuildVRT('', list_tif, VRTNodata=-9999)

    # Saving to TIF
    output = gdal.Translate(
        path_output, temp, format='GTiff', creationOptions=['COMPRESS=LZW'])

    # Creating pyramids
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'LZW')
    output.BuildOverviews('NEAREST', [2, 4, 8, 16, 32])
    output.FlushCache()

    # Success
    print('Successfully saved to %s' % (path_output))


# Converting raster to vector
def raster2vector(path_raster, path_vector, output_format):
    if output_format.lower() == 'kml':
        format = 'KML'
        ext = '.kml'
    elif output_format.lower() == 'geojson':
        format = 'Geojson'
        ext = '.json'
    elif output_format.lower() == 'shp':
        format = 'ESRI Shapefile'
        ext = '.shp'
    else:
        format = 'ESRI Shapefile'
        ext = '.shp'

    # Generate exceptions
    gdal.UseExceptions()

    src_ds = gdal.Open(path_raster)
    num_band = src_ds.RasterCount
    temp = []
    for i in range(1, num_band+1):
        if src_ds is None:
            print('Unable to open %s' % src_fileName)
            sys.exit(1)

        srcband = src_ds.GetRasterBand(i)
        maskband = srcband.GetMaskBand()

        dst_layername = os.path.join(
            path_vector, os.path.splitext(os.path.basename(path_raster))[0] + '_b' + str(i) + ext)
        temp.append(dst_layername)
        dst_fieldname = 'value'

        drv = ogr.GetDriverByName(format)

        # Saving vectors by individual bands
        dst_ds = drv.CreateDataSource(dst_layername)

        source_srs = osr.SpatialReference()
        source_srs.ImportFromWkt(src_ds.GetProjectionRef())

        dst_layer = dst_ds.GetLayerByName(dst_layername)
        dst_layer = dst_ds.CreateLayer(
            dst_layername, geom_type=ogr.wkbPolygon, srs=source_srs)

        fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
        dst_layer.CreateField(fd)
        dst_field = 0
        gdal.Polygonize(srcband, maskband, dst_layer, dst_field, callback=None)
        srcband = None
        src_ds = None
        dst_ds = None
        mask_ds = None
    print('Vector successfully converted to %s' % (dst_layername))
    return temp
