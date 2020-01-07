import os

# Path Inputs
path_input = 'data/Buildings'
path_output = 'model/exp1'

# Model Parameters
num_image_channels = 3
image_size = 200
grid_size = 200
epoch = 20
overlap = 10

# Post Processing
skeletonize_filter = 5
minimum_distance_watershed = 7.5  # 7.5m away buildings

"""
####################### DO NOT CHANGE BEYOND THIS ###########################
"""

# Derived Input paths
path_image = os.path.join(path_input, 'image')
path_label = os.path.join(path_input, 'label')
path_logs = os.path.join(path_output, 'log')
path_tensorboard_log = os.path.join(path_logs, 'tensorboard')

# Path Outputs
path_model = os.path.join(path_output, 'model')
path_accuracy = os.path.join(path_output, 'accuracy')
path_weight = os.path.join(path_output, 'weight')
path_prediction = os.path.join(path_output, 'predicted')
path_tiled = os.path.join(path_output, 'tiled')

# Tiled dataset
path_tiled_image = os.path.join(path_tiled, 'image')
path_tiled_label = os.path.join(path_tiled, 'label')
