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
import logging


# storing geo-referencing information
def getGeodata(imageMap:dict) -> dict:
    """
    Input :
        imageMap: Hashmap of index as key and path of image as value
    Output:
        geoMap: Hashmap of index as key and dict of geographic parameters of the image \
            corresponding to index
    """
    geoMap = {}
    
    # Storing geo-referencing information
    for i, index in enumerate(imageMap.keys()):
        geoMap[index] = {}
        geoTransform, geoProjection, size, _ = read_tif(imageMap[index])
        geoMap[index]['geoTransform'] = geoTransform
        geoMap[index]['geoProjection'] = geoProjection
        geoMap[index]['size'] = size
        geoMap[index]['path'] = imageMap[index]
 
    return geoMap


# Reading raster dataset
def read_tif(path_tif:str):
    """
    Input: TIF image path
    Output: geoTransform, geoProjection, size, arr
    """
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
        no_data = band.GetNoDataValue()
        arr[arr==no_data] = 0
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
        except Exception as e:
            arr = array[:, :]
        arr = cv2.resize(arr, size)
        outdata.GetRasterBand(i+1).WriteArray(arr)
    # outdata.GetRasterBand(1).SetNoDataValue(-9999)##if you want ... \
    # ...\ these values transparent
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
#    logging.info('Gridding Images ...')
#    gridding(args)
    return True


def test_checkres(path:str, size:int, output:str, percent_overlap:float) ->None:
    """
        Gridding all the image present inside given folders.
        Griddings are stored in separate folder of each images named after that image itself
        Input:
            path: input folder containing all the images
            size: size of gridding.
            output: output path of the gridded image
            percent_overlap: Overlapping need to grid large image. Ex - 10
        
        Output:
            None (Output is automatically save to path_output location given)
    """

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
#    logging.info('Gridding Images ...')
#    gridding(args)
    return True


# Saving to JSON
def tojson(dictA, file_json):
    with open(file_json, 'w') as f:
        json.dump(dictA, f, indent=4, separators=(',', ': '),
                  ensure_ascii=False)


# Merging all the tiled tif to single tif using gdalbuildvrt and gdaltranslate
def mergeTile(listTIF: list, path_output:str) ->None:
    """
        Input:
            listTIF: List of path of TIF files to be nerged
            path_ouput: file location of mergerd data to be saved
        Output:
            None (Output is automatically save to path_output location given)
    """
    # Building VRT
    temp = gdal.BuildVRT('', listTIF, VRTNodata=-9999)

    # Saving to TIF
    output = gdal.Translate(
        path_output, temp, format='GTiff', creationOptions=['COMPRESS=LZW'])

    # Creating pyramids
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'LZW')
    output.BuildOverviews('NEAREST', [2, 4, 8, 16, 32])
    output.FlushCache()

    # Success
    logging.info('Successfully saved to %s' % (path_output))


# Converting raster to vector
def raster2vector(path_raster: str, path_output: str, output_format='shp') ->list:
    """
        Input:
            path_raster: Input path of TIF to be processed
            path_output: Ouput path of TIF to be saved
            output_format: Format of vector data to be saved
        Output: 
            list of vector paths
    """
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
    if src_ds is None:
        logging.error('Unable to open %s' % path_raster)
        sys.exit(1)
    
    num_band = src_ds.RasterCount
    temp = []
    for i in range(1, num_band+1):
        
        srcband = src_ds.GetRasterBand(i)
        maskband = srcband.GetMaskBand()

        dst_layername = os.path.join(
            path_output, os.path.splitext(os.path.basename(path_raster))[0] + '_b' + str(i) + ext)
        
        checkdir(os.path.dirname(dst_layername))
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
        
        if dst_layer is None:
            logging.error('Unale to create vector dataset of format: {}'.format(dst_layername))
        
        fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)

        dst_layer.CreateField(fd)
        dst_field = 0
        gdal.Polygonize(srcband, maskband, dst_layer, dst_field, callback=None)
    
        srcband = None
        src_ds = None
        dst_ds = None
        mask_ds = None
        
    logging.info('Vector successfully converted to %s' % (dst_layername))
    return temp
