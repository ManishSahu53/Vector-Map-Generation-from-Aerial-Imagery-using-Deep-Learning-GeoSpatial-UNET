""" Model is defined here"""
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, UpSampling2D, ZeroPadding2D
from keras.models import Model


def unet(image_size):
    inputs = Input((image_size, image_size, 3))
    print (inputs.shape)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    print (conv1.shape)
    bn1 = BatchNormalization()(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(bn1)
    print (conv2.shape)
    bn2 = BatchNormalization()(conv2)

    pool1 = MaxPooling2D(pool_size=(2, 2))(bn2)
    print (pool1.shape)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    print (conv3.shape)
    bn3 = BatchNormalization()(conv3)

    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn3)
    print (conv4.shape)
    bn4 = BatchNormalization()(conv4)

    pool2 = MaxPooling2D(pool_size=(2, 2))(bn4)
    print (pool2.shape)

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    print (conv5.shape)
    bn5 = BatchNormalization()(conv5)

    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(bn5)
    print (conv6.shape)
    bn6 = BatchNormalization()(conv6)

    pool3 = MaxPooling2D(pool_size=(2, 2))(bn6)
    print (pool3.shape)

    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    print (conv7.shape)
    bn7 = BatchNormalization()(conv7)

    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(bn7)
    print (conv8.shape)
    bn8 = BatchNormalization()(conv8)

    pool4 = MaxPooling2D(pool_size=(2, 2))(bn8)
    print (pool4.shape)

    conv9 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    print (conv9.shape)
    bn9 = BatchNormalization()(conv9)

    conv10 = Conv2D(512, (3, 3), activation='relu', padding='same')(bn9)
    print (conv10.shape)
    bn10 = BatchNormalization()(conv10)

    _up1 = concatenate([Conv2DTranspose(256, (3, 3), strides=(
        2, 2), padding='valid')(bn10), conv8], axis=3)
    print("Near neighbour upsampling " + str(_up1.shape))

    conv11 = Conv2D(256, (3, 3), activation='relu', padding='same')(_up1)
    print (conv11.shape)
    bn11 = BatchNormalization()(conv11)

    conv12 = Conv2D(256, (3, 3), activation='relu', padding='same')(bn11)
    print (conv12.shape)
    bn12 = BatchNormalization()(conv12)

    _up2 = concatenate([Conv2DTranspose(128, (3, 3), strides=(
        2, 2), padding='same')(bn12), conv6], axis=3)
    print("Near neighbour upsampling " + str(_up2.shape))

    conv13 = Conv2D(128, (3, 3), activation='relu', padding='same')(_up2)
    print (conv13.shape)
    bn13 = BatchNormalization()(conv13)

    conv14 = Conv2D(128, (3, 3), activation='relu', padding='same')(bn13)
    print (conv14.shape)
    bn14 = BatchNormalization()(conv14)

    _up3 = concatenate([Conv2DTranspose(64, (2, 2), strides=(
        2, 2), padding='same')(bn14), conv4], axis=3)
    print("Near neighbour upsampling " + str(_up3.shape))

    conv15 = Conv2D(64, (3, 3), activation='relu', padding='same')(_up3)
    print (conv15.shape)
    bn15 = BatchNormalization()(conv15)

    conv16 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn15)
    print (conv16.shape)
    bn16 = BatchNormalization()(conv16)

    _up4 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(bn16), conv2], axis=3)
    print("Near neighbour upsampling " + str(_up4.shape))

    conv17 = Conv2D(32, (3, 3), activation='relu', padding='same')(_up4)
    print (conv17.shape)
    bn17 = BatchNormalization()(conv17)

    conv18 = Conv2D(32, (3, 3), activation='relu', padding='same')(bn17)
    print (conv18.shape)
    bn18 = BatchNormalization()(conv18)

    conv19 = Conv2D(1, (1, 1), activation='sigmoid')(bn18)
    print (conv19.shape)

    model = Model(inputs=[inputs], outputs=[conv19])
    return model


def stan_unet(image_size):
    inputs = Input((image_size, image_size, 3))
    print (inputs.shape)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    print (conv1.shape)
    # bn1 = BatchNormalization()(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    print (conv2.shape)
    # bn2 = BatchNormalization()(conv2)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print (pool1.shape)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    print (conv3.shape)
    # bn3 = BatchNormalization()(conv3)

    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    print (conv4.shape)
    # bn4 = BatchNormalization()(conv4)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    print (pool2.shape)

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    print (conv5.shape)
    # bn5 = BatchNormalization()(conv5)

    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    print (conv6.shape)
    # bn6 = BatchNormalization()(conv6)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)
    print (pool3.shape)

    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    print (conv7.shape)
    # bn7 = BatchNormalization()(conv7)

    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    print (conv8.shape)
    # bn8 = BatchNormalization()(conv8)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)
    print (pool4.shape)

    conv9 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    print (conv9.shape)
    # bn9 = BatchNormalization()(conv9)

    conv10 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv9)
    print (conv10.shape)
    # bn10 = BatchNormalization()(conv10)

    _up1 = concatenate([Conv2DTranspose(256, (3, 3), strides=(
        2, 2), padding='valid')(conv10), conv8], axis=3)
    print("Near neighbour upsampling " + str(_up1.shape))

    conv11 = Conv2D(256, (3, 3), activation='relu', padding='same')(_up1)
    print (conv11.shape)
    # bn11 = BatchNormalization()(conv11)

    conv12 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv11)
    print (conv12.shape)
    # bn12 = BatchNormalization()(conv12)

    _up2 = concatenate([Conv2DTranspose(128, (3, 3), strides=(
        2, 2), padding='same')(conv12), conv6], axis=3)
    print("Near neighbour upsampling " + str(_up2.shape))

    conv13 = Conv2D(128, (3, 3), activation='relu', padding='same')(_up2)
    print (conv13.shape)
    # bn13 = BatchNormalization()(conv13)

    conv14 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv13)
    print (conv14.shape)
    # bn14 = BatchNormalization()(conv14)

    _up3 = concatenate([Conv2DTranspose(64, (2, 2), strides=(
        2, 2), padding='same')(conv14), conv4], axis=3)
    print("Near neighbour upsampling " + str(_up3.shape))

    conv15 = Conv2D(64, (3, 3), activation='relu', padding='same')(_up3)
    print (conv15.shape)
    # bn15 = BatchNormalization()(conv15)

    conv16 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv15)
    print (conv16.shape)
    # bn16 = BatchNormalization()(conv16)

    _up4 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(conv16), conv2], axis=3)
    print("Near neighbour upsampling " + str(_up4.shape))

    conv17 = Conv2D(32, (3, 3), activation='relu', padding='same')(_up4)
    print (conv17.shape)
    # bn17 = BatchNormalization()(conv17)

    conv18 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv17)
    print (conv18.shape)
    # bn18 = BatchNormalization()(conv18)

    conv19 = Conv2D(1, (1, 1), activation='sigmoid')(conv18)
    print (conv19.shape)

    model = Model(inputs=[inputs], outputs=[conv19])
    return model
