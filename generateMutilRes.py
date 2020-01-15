import gdal
import os
import config
from src import util
import logging

util.check_dir(config.path_logs)
util.set_logger(os.path.join(config.path_logs, 'generateMutiRes.log'))


def drange(start: float, stop: float, step: float):
    r = start
    while r < stop:
        yield round(r, 1)
        r += step


def getRes(path_tif: str) -> None:
    """
    Reading Resolution of input iamge
    Input:
        path_tif: path of input to be processed
    Output:
        None
    """
    ds = gdal.Open(path_tif)
    resolution = round(ds.GetGeoTransform()[1], 1)
    return resolution


logging.info('Iterating Training data in folder: {}'.format(config.path_image))

for root, dirs, files in os.walk(config.path_image):
    for file in files:
        if file.endswith(tuple(config.image_ext)):

            currentResolution = getRes(os.path.join(root, file))

            listResolution = list(drange(
                max(currentResolution, config.minResolution),
                config.maxResolution,
                config.step))

            for res in listResolution:
                gdalOption = gdal.WarpOptions(
                    format='VRT', xRes=res, yRes=res)

                # Creating output file name
                path_image_output = os.path.join(
                    config.path_image_vrt,
                    util.getNamenoExt(file),
                    util.getNamenoExt(file) + '_' + str(res)+'.vrt')

                path_label_output = os.path.join(
                    config.path_label_vrt,
                    util.getNamenoExt(file),
                    util.getNamenoExt(file) + '_' + str(res)+'.vrt')

                # Creating folders
                util.check_dir(os.path.dirname(path_image_output))
                util.check_dir(os.path.dirname(path_label_output))

                logging.info('Processing : {}'.format(os.path.join(
                    root, file)))
                # Creating VRT of input image
                gdal.Warp(path_image_output, os.path.abspath(os.path.join(
                    root, file)), options=gdalOption)

                logging.info('Processing : {}'.format(os.path.join(
                    config.path_label, file)))
                # Creating VRT of label image
                gdal.Warp(path_label_output,  os.path.abspath(os.path.join(
                    config.path_label, file)), options=gdalOption)


logging.info('Iterating Validation data in folder: {}'.format(config.path_vali_image))

for root, dirs, files in os.walk(config.path_vali_image):
    for file in files:
        if file.endswith(tuple(config.image_ext)):

            currentResolution = getRes(os.path.join(root, file))

            listResolution = list(drange(
                max(currentResolution, config.minResolution),
                config.maxResolution,
                config.step))

            for res in listResolution:
                gdalOption = gdal.WarpOptions(
                    format='VRT', xRes=res, yRes=res)

                # Creating output file name
                path_vali_image_output = os.path.join(
                    config.path_vali_image_vrt,
                    util.getNamenoExt(file),
                    util.getNamenoExt(file) + '_' + str(res)+'.vrt')

                path_vali_label_output = os.path.join(
                    config.path_vali_label_vrt,
                    util.getNamenoExt(file),
                    util.getNamenoExt(file) + '_' + str(res)+'.vrt')

                # Creating folders
                util.check_dir(os.path.dirname(path_vali_image_output))
                util.check_dir(os.path.dirname(path_vali_label_output))

                logging.info('Processing : {}'.format(os.path.join(
                    root, file)))
                # Creating VRT of input image
                gdal.Warp(path_vali_image_output, os.path.abspath(os.path.join(
                    root, file)), options=gdalOption)

                logging.info('Processing : {}'.format(os.path.join(
                    config.path_label, file)))
                # Creating VRT of label image
                gdal.Warp(path_vali_label_output,  os.path.abspath(os.path.join(
                    config.path_vali_label, file)), options=gdalOption)
