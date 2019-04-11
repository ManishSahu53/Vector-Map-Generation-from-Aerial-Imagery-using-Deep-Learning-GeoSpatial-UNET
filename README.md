# Unet Implementation is Keras with GPU

## Background
I have created this repo to do instance segmentation uisng U-NET and simple post processing techniques for large geospatial dtaset.
Satellite or UAV generated dataset have special charecteristics:

1. They are geo-referenced. That means easch pixel have a geo-coordinate (latitude and longitude) 
2. They are generally of very large size > 10,000 x 10,000 pixels

Thus some preprocessing and post processing is needed in order to train the Deep network with such large dataset.

## Abstract
We propose a simple yet efficient technique to leverage semantic segmentation model to extract and separate individual buildings in densely compacted areas using medium resolution satellite/UAV orthoimages. We adopted standard UNET architecture, additionally added batch normalization layer after every convolution, to label every pixel in the image. The result obtained is fed into proposed post-processing pipeline for separating connected binary blobs of buildings and converting it into GIS layer for further analysis as well as for generating 3D buildings. The proposed algorithm extracts building footprints from aerial images, transform semantic to instance map and convert it into GIS layers to generate 3D buildings. We integrated this method in Indshine’s cloud platform to speed up the process of digitization, generate automatic 3D models, and perform the geospatial analysis. Our network achieved ~70% Dice coefficient for the segmentation process.

## Introduction
One of the major challenges in the GIS industry is the extraction of urban feature objects like buildings, roads, trees, etc. from Satellite and UAV images. Moreover, feature extraction is never an end product of any study rather it acts as an intermediate data from which analyses are done, so there is a need for a mechanism or a platform which can provide/generate data on the fly at industrial standard formats.

## Methodology
### Semantic segmentation

For semantic segmentation of building footprint from RGB images, we used U-Net architecture. We added an extra batch normalization layer after every convolution layer to avoid any activations to take extreme values, to reduce sensitivity towards initial weights initialization and reduce overfitting of the model. (Olaf Ronneberger, 2015) describes all the details of architecture and used it for medical image segmentation. In the original U- Net, Olaf used 512x512 image size in the network but here we have used 200x200 image size. 

![Semantic Segmentation issue](https://github.com/ManishSahu53/geospatial_unet/blob/master/images/Connected%20Blobs.png)


### Noise Removal
For noise removal, we used basic morphological operation on binary data. We first applied erosion followed by dilation followed by thresholding with respect to area.

### Distance Transform
To separate connected buildings, we assumed that connections/binary bridge in the joined binary blobs is less than the area of buildings itself. We used this fact and applied distance transformation (Jain, 1989) to binary image. Thus connection/binary bridge were assigned less weight as compared to buildings itself. See Figure below for illustration.

![Distance Transformation](https://github.com/ManishSahu53/geospatial_unet/blob/master/images/Distance%20Transform.png)

### Local Maxima
To overcome the problem described above, we used Local Maxima approach. Since connections are smaller than the buildings itself, finding local maxima ensured that it lies inside building area and not in the connection. This local maximum point acted as input source/sink to the watershed algorithm.

### Watershed Segmentation
Watershed segmentation is quite popular for image segmentation. Local Maxima obtained acted here as sink point and negative of distance transform as cost map. This helped to separate connected binary blobs effectively. Each blob is given a unique index for further processing

![Watershed Segmentation](https://github.com/ManishSahu53/geospatial_unet/blob/master/images/Watershed.png)

### Vectorization
Raster obtained from the watershed segmentation is vectorized using gdal/ogr library (Contributors of GDAL/OGR, 2018). Spatial references and coordinate systems are preserved at every step and are transferred to vector file for correct overlaying. While converting from raster to vector, the output has a lot of vertices and noises. Vector is then simplified using Douglas- Peucker algorithm (David Douglas, 1973). This helps to preserve the overall geometry of the shape while simplifying number of vertices. Basic attributes like area, perimeter and elevation of the buildings were automatically added. Finally, minimum bounding box was used and saved

![Vectorization](https://github.com/ManishSahu53/geospatial_unet/blob/master/images/BoundingBox.png)


### RESULTS
1. ![Small_village_in_Maharashtra_India](https://github.com/ManishSahu53/geospatial_unet/blob/master/images/LC.png)
2. ![Dense_populated_area_in_Africa](https://github.com/ManishSahu53/geospatial_unet/blob/master/images/Africa.png)
3. ![Planned_colony](https://github.com/ManishSahu53/geospatial_unet/blob/master/images/Planned.png)
4. ![Slums](https://github.com/ManishSahu53/geospatial_unet/blob/master/images/Slums.png)

## How to use
```
python3 train.py [-h] -d DATA [-s SIZE] [-c CLASSES] [-sg SKIP_GRIDDING]
```

See description below to see all available options
optional arguments:
```
  -h, --help            show this help message and exit
  -d DATA, --data DATA  Input directory containing images and labels
  -s SIZE, --size SIZE  Input size of image to be used. [Default] = 200
  -c CLASSES, --classes CLASSES
                        Input number of classes.[Default] = 1
  -sg SKIP_GRIDDING, --skip_gridding SKIP_GRIDDING
                        If gridding is already done then skip it. [Default] is
                        No = 0
```
