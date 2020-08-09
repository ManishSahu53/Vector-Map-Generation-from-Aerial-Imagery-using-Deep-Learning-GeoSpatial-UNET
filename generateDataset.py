import logging
import config
import os
from src import io, util


util.check_dir(config.path_logs)
util.set_logger(os.path.join(config.path_logs, 'generateDataset.log'))


# Checking directories
util.check_dir(config.path_model)
util.check_dir(config.path_weight)
util.check_dir(config.path_prediction)
util.check_dir(config.path_tiled)
util.check_dir(config.path_tiled_image)
util.check_dir(config.path_tiled_label)

# Validation
util.check_dir(config.path_vali_tiled_image)
util.check_dir(config.path_vali_tiled_label)

path_image = config.path_image_vrt
path_label = config.path_label_vrt

path_vali_image = config.path_vali_image_vrt
path_vali_label = config.path_vali_label_vrt

logging.info('path_image: {}, path_label: {}, path_vali_image: {}, path_vali_label: {}'.format(
    path_image, path_label, path_vali_image, path_vali_label))


logging.info('Tiling Training Images...')
tile_image = io.checkres(path_image, config.grid_size,
                         config.path_tiled_image, config.overlap)

logging.info('Tiling Training Labels...')
tile_label = io.checkres(path_label, config.grid_size,
                         config.path_tiled_label, config.overlap)

logging.info('Tiling Training data Completed')

logging.info('Tiling Validation Images...')
tile_image = io.checkres(path_vali_image, config.grid_size,
                         config.path_vali_tiled_image, config.overlap)

logging.info('Tiling Validation Labels...')
tile_label = io.checkres(path_vali_label, config.grid_size,
                         config.path_vali_tiled_label, config.overlap)

logging.info('Tiling Validation data Completed')
