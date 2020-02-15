import os
"""
backbone = {'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152', 'MobileNetV1', 'MobileNetV2', 'Xception', 'Xception-DeepLab'}
model = {'FCN', 'PSPNet', 'SegNet', 'UNet', 'PAN', 'DeepLabV3', 'DeepLabV3Plus', 'RefineNet', 'DenseASPP', 'BiSegNet'}          
"""
# Path Inputs
path_input = 'data/Buildings/train'
# Inside this folder, should contain an image and label folder \
# Example - train/image, train/label

path_output = 'model/DenseASPP'
path_validation = 'data/Buildings/test'
path_vali_output = os.path.join(path_output, 'validation')

# generateDatset Parameters
minResolution = 0.1  # meters
maxResolution = 1  # meters
step = 0.1  # meters

# Model Parameters
modelArchitecture = 'DenseASPP'
modelBackone = 'ResNet50'
num_image_channels = 3
num_label_channels = 1
validation_split = 0.15
image_size = 256
grid_size = 256
epoch = 30
overlap = 10
batch = 8
learningRate = 1e-4
image_ext = ['.tif', '.tiff', '.vrt']

# Post Processing
skeletonize_filter = 5
minimum_distance_watershed = 7.5  # 7.5m away buildings
simplify_parameter = 0.7  # 0.7m can be adjusted while simplifying vectors
erosion_filter = 3
watershed_neighbour = 4  # 4 or 8

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
path_vrt = os.path.join(path_output, 'vrt')
path_image_vrt = os.path.join(path_vrt, 'image')
path_label_vrt = os.path.join(path_vrt, 'label')

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

path_vali_vrt = os.path.join(path_vali_output, 'vrt')
path_vali_image_vrt = os.path.join(path_vali_vrt, 'image')
path_vali_label_vrt = os.path.join(path_vali_vrt, 'label')
