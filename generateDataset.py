import logging
import config
import os
from src import io, util


util.check_dir(config.path_logs)
util.set_logger(os.path.join(config.path_logs, 'generateDataset.log'))

logging.info('Tiling Images...')

# Checking directories
util.check_dir(config.path_model)
util.check_dir(config.path_weight)
util.check_dir(config.path_prediction)
util.check_dir(config.path_tiled)
util.check_dir(config.path_tiled_image)
util.check_dir(config.path_tiled_label)

tile_image = io.checkres(config.path_image, config.grid_size,
                         config.path_tiled_image, config.overlap)

logging.info('Tiling Labels...')
tile_label = io.checkres(config.path_label, config.grid_size,
                         config.path_tiled_image, config.overlap)

logging.info('Tiling Completed')
