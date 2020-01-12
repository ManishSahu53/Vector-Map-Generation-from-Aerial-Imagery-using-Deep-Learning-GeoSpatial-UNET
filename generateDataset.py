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


logging.info('Tiling Training Images...')
tile_image = io.checkres(config.path_image, config.grid_size,
                         config.path_tiled_image, config.overlap)

logging.info('Tiling Training Labels...')
tile_label = io.checkres(config.path_label, config.grid_size,
                         config.path_tiled_label, config.overlap)

logging.info('Tiling Training data Completed')

logging.info('Tiling Validation Images...')
tile_image = io.checkres(config.path_vali_image, config.grid_size,
                         config.path_vali_tiled_image, config.overlap)

logging.info('Tiling Validation Labels...')
tile_label = io.checkres(config.path_vali_label, config.grid_size,
                         config.path_vali_tiled_label, config.overlap)

logging.info('Tiling Validation data Completed')
