"""Post Processing of vector and Raster dataset"""
import shapefile
import numpy as np
from scipy.spatial import ConvexHull
import fiona
from fiona.crs import from_epsg
from shapely.geometry import mapping, Polygon
from shapely.geometry import MultiPolygon
import os
from pyproj import Proj, transform
import cv2
from src import io
import gdal
import ogr
import osr

# For Post Processing libraries
from scipy.ndimage.interpolation import rotate
from skimage.morphology import skeletonize as skt
# from skimage.morphology import medial_axis
from skimage.feature import peak_local_max
from skimage.morphology import watershed, remove_small_objects
from scipy import ndimage as ndi


# Mininum bounding box/ axis aligned bounding box
def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices

    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    #     rotations = np.vstack([
    #         np.cos(angles),
    #         -np.sin(angles),
    #         np.sin(angles),
    #         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


# Calculating area of polygon
def polygon_area(x, y):
    correction = x[-1] * y[0] - y[-1] * x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return round(0.5*np.abs(main_area + correction), 2)


# Calculating area of bounding box
def aabbox(path_shp, path_output):

    sf = shapefile.Reader(path_shp)

    # Reading in Fiona
    fio = fiona.open(path_shp, 'r')

    # Extracting coordinates
    coord_sys = fio.crs_wkt

    shapes = sf.shapes()
    num_shp = len(shapes)
    poly = []
    area = []

    # Getting records
    rec = sf.records()

    for n in range(num_shp):
        if rec[n][0] > 0:  # Removing '0' background value
            coord = np.asarray(shapes[n].points)
            bbox = np.asarray(minimum_bounding_rectangle(coord))
            area.append(polygon_area(bbox[:, 0], bbox[:, 1]))
            b = np.append(bbox, [bbox[0]], axis=0)
            poly.append(Polygon(b))

    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int',
                       'Area': 'float'},
    }

    # Write a new Shapefile
    with fiona.open(path_output, 'w', crs=coord_sys, driver='ESRI Shapefile', schema=schema) as c:
        # If there are multiple geometries, put the "for" loop here
        for i in range(len(poly)):
            c.write({
                'geometry': mapping(Polygon(poly[i])),
                'properties': {'id': i,
                               'Area': area[i]},
            })


# Image erosion (N filter) and dilation (N Filter)
def erosion(path_image, filter, path_output):

    erode_kernel = np.ones((filter, filter), np.uint8)
    geotransform, geoprojection, size, arr = io.read_tif(path_image)
    # Image erosion
    erode = cv2.erode(arr, erode_kernel)

    # Image dilation
    dilate_kernel = np.ones((filter, filter), np.uint8)
    dilate = cv2.dilate(erode, dilate_kernel)

    # removing smaller pixels
    cell_size = geotransform[1]
    min_area = 9  # 9 sq.metres
    num_pixel = int(min_area/cell_size*cell_size)
    dilate = np.asarray(dilate, dtype=int)
    dilate = np.absolute(dilate)
    cleaned = remove_small_objects(dilate, min_size=num_pixel, connectivity=2)

    print('Saving erosion to %s' % (path_output))
    io.write_tif(path_output, cleaned, geotransform, geoprojection, size)
    return path_output


# Skeletonize raster dataset
def skeletonize(path_image, path_output):
    filter = 5

    geotransform, geoprojection, size, arr = io.read_tif(path_image)
    """Array input must be binary
    Output array is also binary
    """
    arr[arr > 0] = 1
    dilate_kernel = np.ones((filter, filter), np.uint8)
    arr = cv2.dilate(arr, dilate_kernel)
    skeleton = skt(arr)

    print('Saving skeleton to %s' % (path_output))
    io.write_tif(path_output, skeleton*255, geotransform, geoprojection, size)
    return path_output


# Watershed Segmentation
def waterseg(path_image, filter, path_output):

    geotransform, geoprojection, size, array = io.read_tif(path_image)
    """ Minimum distance between two objects is 5m. 
    distance = 5/cell_size
    """
    dim_array = array.shape
    if len(dim_array) > 2:
        depth = dim_array[2]
    else:
        depth = 1

    labels = np.zeros(array.shape)
    for i in range(depth):
        try:
            arr = array[:, :, i]
        except:
            arr = array[:, :]
        distance = int(7.5/geotransform[1])
        D = ndi.distance_transform_edt(arr)
        localMax = peak_local_max(
            D, indices=False, min_distance=distance, labels=arr)

        # 4 Connected pixels, we can also use 8 connected pixels
        if int(filter) == 4:
            filter = [[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]]
        elif int(filter) == 8:
            filter = [[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]]

        filter = np.asarray(filter)

        # markers = ndimage.label(localMax, structure=filter)[0]
        markers = ndi.label(localMax, structure=filter)[0]
        try:
            labels[:, :, i] = watershed(-D, markers, mask=arr)
        except:
            labels = watershed(-D, markers, mask=arr)
    print('Saving watershed segmentation to %s' % (path_output))
    io.write_tif(path_output, labels, geotransform, geoprojection, size)
    return path_output


# Calculating area of bounding box
def simplify_polygon(path_shp, para, path_output):
    gdal.UseExceptions()
    ogr.UseExceptions()

    def addPolygon(simplePolygon, dst_layer, index):
        featureDefn = dst_layer.GetLayerDefn()
        polygon = ogr.CreateGeometryFromWkb(simplePolygon)
        dst_feat = ogr.Feature(featureDefn)
        dst_feat.SetGeometry(polygon)

        geom = dst_feat.GetGeometryRef()
        dst_feat.SetField('id', index)
        dst_feat.SetField('area', geom.Area())

        dst_layer.CreateFeature(dst_feat)
        dst_layer.SyncToDisk()

    def multipoly2poly(src_lyr, para, dst_layer):
        count = 0
        for src_feat in src_lyr:
            if src_feat.GetField(0) > 0:
                count = count + 1
                geom = src_feat.GetGeometryRef()
                if geom.GetGeometryName() == 'MULTIPOLYGON':
                    for geom_part in geom:
                        x = geom_part.SimplifyPreserveTopology(para)
                        addPolygon(x.ExportToWkb(), dst_layer, count)
                else:
                    x = geom.SimplifyPreserveTopology(para)
                    addPolygon(x.ExportToWkb(), dst_layer, count)
            else:
                continue

    # Reading source shp
    drv = ogr.GetDriverByName('ESRI Shapefile')
    src_ds = drv.Open(path_shp, 0)
    src_lyr = src_ds.GetLayer()

    # Reading source coordinate system
    src_srs = osr.SpatialReference()

    # from Layer
    spatialRef = src_lyr.GetSpatialRef()
    # from Geometry
    feature = src_lyr.GetNextFeature()
    geom = feature.GetGeometryRef()
    spatialRef = geom.GetSpatialReference()
    src_srs.ImportFromWkt(spatialRef.ExportToWkt())

    # Creating destination shp
    dst_ds = drv.CreateDataSource(path_output)
    dst_layer = dst_ds.GetLayerByName(path_output)
    dst_layer = dst_ds.CreateLayer(
        path_output, geom_type=ogr.wkbPolygon, srs=src_srs)

    # Add an ID and area field
    idField = ogr.FieldDefn('id', ogr.OFTInteger)
    areaField = ogr.FieldDefn('area', ogr.OFTReal)
    dst_layer.CreateField(idField)
    dst_layer.CreateField(areaField)

    # Simplification of polygons
    multipoly2poly(src_lyr, para, dst_layer)
