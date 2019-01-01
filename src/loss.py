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
    axes = (1, 2, 3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)

    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)


def jaccard_coef(y_true, y_pred, smooth=10.0):
    '''Average jaccard coefficient per batch.'''
    axes = (1, 2, 3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)
