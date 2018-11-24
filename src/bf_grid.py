import os
import subprocess
import time


def bf_grid(image_lo, image_size1, image_size2, overlap, tile_image_lo):
    gridding = 'bf_gridding.py'

    extension1 = 'tif'
    extension2 = 'tiff'
    i = 1
    total = 0

    for file in os.listdir(image_lo):
        if file.endswith(extension1):
            total = total + 1

    for file in os.listdir(image_lo):
        if file.endswith(extension1):
            i = i+1
            st_time = time.time()
            pFeatures = subprocess.call(['python3', gridding, 
                                         '-ps', str(image_size1),str(image_size2),
                                         '-overlap', str(overlap),
                                         '-targetDir', tile_image_lo,
                                         os.path.join(image_lo, file)])
            end_time = time.time()
            print(end_time - st_time)
            print("Completed " + str(i) + " Out of " + str(total) + "files")

        elif file.endswith(extension2):
            i = i+1
            st_time = time.time()
            pFeatures = subprocess.call(['python3', gridding, 
                                         '-ps', str(image_size1), str(image_size2), 
                                         '-overlap', str(overlap),
                                         '-targetDir', tile_image_lo, 
                                         os.path.join(image_lo, file)])
            end_time = time.time()
            print(end_time - st_time)
            print("Completed " + str(i) + " Out of " + str(total) + "files")
