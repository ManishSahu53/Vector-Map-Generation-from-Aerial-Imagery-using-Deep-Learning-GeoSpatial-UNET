from keras import backend as K
import tensorflow as tf
from scipy import ndimage as ndi
import keras


def weighted_binary_crossentropy(y_true, y_pred):
    weight = y_true[:, :, :, -1]
    y_true = y_true[:, :, :, 0]

    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    # Scaling Distance Transform X 10 since values are too small
    loss = K.binary_crossentropy(y_true, y_pred[:, :, 0])
    loss = loss * weight
    return loss
    # keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
