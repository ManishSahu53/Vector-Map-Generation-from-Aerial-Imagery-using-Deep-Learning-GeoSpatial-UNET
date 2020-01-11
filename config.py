import os

# Path Inputs
path_input = 'data/Buildings/train'
path_output = 'model/exp1'
path_validation = 'data/Buildings/test'
path_vali_output = os.path.join(path_output, 'validation')

# Model Parameters
num_image_channels = 3
num_label_channels = 1
validation_split = 0.15
image_size = 200
grid_size = 200
epoch = 20
overlap = 10
batch = 32
image_ext1 = '.tif'
image_ext2 = '.tiff'

# Post Processing
skeletonize_filter = 5
minimum_distance_watershed = 7.5  # 7.5m away buildings

"""
####################### DO NOT CHANGE BEYOND THIS ###########################
"""
# Configuration file
path_configuration = os.path.join(path_output, 'param.json')

# Derived Input paths
path_image = os.path.join(path_input, 'image')
path_label = os.path.join(path_input, 'label')
path_logs = os.path.join(path_output, 'log')
path_tensorboard_log = os.path.join(path_logs, 'tensorboard')

# Path Outputs
path_model = os.path.join(path_output, 'model')
path_weight = os.path.join(path_output, 'weight')
path_prediction = os.path.join(path_output, 'predicted')
path_tiled = os.path.join(path_output, 'tiled')

# Tiled Training dataset
path_tiled_image = os.path.join(path_tiled, 'image')
path_tiled_label = os.path.join(path_tiled, 'label')

# Derived Validation paths
path_vali_image = os.path.join(path_validation, 'image')
path_vali_label = os.path.join(path_validation, 'label')
path_vali_tiled = os.path.join(path_vali_output, 'tiled')

# Tiled Validation dataset
path_vali_tiled_image = os.path.join(path_vali_tiled, 'image')
path_vali_tiled_label = os.path.join(path_vali_tiled, 'label')

