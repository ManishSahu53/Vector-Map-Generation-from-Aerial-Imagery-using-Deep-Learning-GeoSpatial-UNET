""" Defining loss functions"""
import numpy as np
from keras import backend as K

"""
Multiclass loss
https://stats.stackexchange.com/questions/255465/accuracy-vs-jaccard-for-multiclass-problem/256140
"""


def jaccard_distance(y_true, y_pred):
    smooth = 1
    """
    https://en.wikipedia.org/wiki/Jaccard_index
    """
    intersection = np.sum(y_true * y_pred)
    sum_ = np.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dice_coef(y_true, y_pred, smooth=10.0):
    '''Average dice coefficient per batch.'''
    dim_true = y_true.shape
    dim_pred = y_pred.shape

    if dim_true != dim_pred:
        print('Dimension of GroundTruth and prediction does not match')
        sys.exit()

    else:
        depth = len(dim_true)

    if np.max(y_pred) > 1:
        y_pred = y_pred / 255.0
    if np.max(y_true) > 1:
        y_true = y_true / 255.0

    intersection = np.sum(np.multiply(y_true, y_pred))
    summation = np.sum(y_true) + np.sum(y_pred)

    return np.mean((2.0 * intersection + smooth) / (summation + smooth))


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)


def jaccard_coef(y_true, y_pred, smooth=10.0):
    '''Average jaccard coefficient per batch.'''
    dim_true = y_true.shape
    dim_pred = y_pred.shape

    if dim_true != dim_pred:
        print('Dimension of GroundTruth and prediction does not match')
        sys.exit()

    else:
        depth = len(dim_true)

    if np.max(y_pred) > 1:
        y_pred = y_pred / 255.0
    if np.max(y_true) > 1:
        y_true = y_true / 255.0

    intersection = np.sum(np.multiply(y_true, y_pred))
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return np.mean((intersection + smooth) / (union + smooth))
