import os
import subprocess
import time
import logging
import config


# Checking and creating directory in the path
def checkdir(path):
    if not os.path.exists(path):
        os.makedirs(os.path.abspath(path))


def bf_grid(image_lo, image_size1, image_size2, overlap, tile_image_lo):
    gridding = 'src/bf_gridding.py'

    i = 1
    total = 0

    
    for root, dirs, files in os.walk(image_lo):
        for file in files:
            if file.endswith(tuple(config.image_ext)):
                total = total + 1
            
    for root, dirs, files in os.walk(image_lo):
        for file in files:
            if file.endswith(tuple(config.image_ext)):
                i = i+1
                st_time = time.time()
                pFeatures = subprocess.call(['python3', gridding,
                                            '-ps', str(image_size1),
                                            str(image_size2),
                                            '-overlap', str(overlap),
                                            '-targetDir', tile_image_lo,
                                            os.path.join(root, file)])
                end_time = time.time()
                logging.info(end_time - st_time)
                logging.info("Completed " + str(i-1) +
                            " Out of " + str(total) + " files")


def test_grid(image_lo, image_size1, image_size2, overlap, tile_image_lo):
    gridding = 'src/bf_gridding.py'


    i = 1
    total = 0

    for root, dirs, files in os.walk(image_lo):
        for file in files:
            if file.endswith(tuple(config.image_ext)):
                total = total + 1

    for root, dirs, files in os.walk(image_lo):
        for file in files:
            if file.endswith(tuple(config.image_ext)):
                i = i+1
                targetDir = os.path.join(tile_image_lo, os.path.splitext(file)[0])
                checkdir(targetDir)
                st_time = time.time()
                pFeatures = subprocess.call(['python3', gridding,
                                            '-ps', str(image_size1),
                                            str(image_size2),
                                            '-overlap', str(overlap),
                                            '-targetDir', targetDir,
                                            os.path.join(root, file)])
                end_time = time.time()
                logging.info(end_time - st_time)
                logging.info("Completed " + str(i-1) +
                            " Out of " + str(total) + " files")


# Gridding Individual TIF file
def grid_file(path_data, path_output):
    """
        Input : 
            path_data : path of TIF file to be gridded
            path_output : output location of gridded file
        Return : Nothing. Output is automatically written into path_output
    """
    gridding = 'src/bf_gridding.py'

    pFeatures = subprocess.call(['python3', gridding,
                                 '-ps', str(config.image_size),
                                 str(config.image_size),
                                 '-overlap', str(config.overlap),
                                 '-targetDir', path_output,
                                 path_data])
    