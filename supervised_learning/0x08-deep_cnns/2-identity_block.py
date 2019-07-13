#!/usr/bin/env python3
"""
Builds an identity block
Skip conenction/shortcut
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Returns: the activated output of the identity block
    """
    conv_layer1 = K.layers.Conv2D(filters[0], (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(A_prev)
    conv_layer1 = K.layers.BatchNormalization()(conv_layer1)

    conv_layer2 =  K.layers.Conv2D(filters[1], (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_layer1)
    conv_layer2 = K.layers.BatchNormalization()(conv_layer2)

    conv_layer3 =  K.layers.Conv2D(filters[2], (1,1), activation='relu',  padding='same', kernel_initializer='he_normal')(conv_layer2)
    conv_layer3 = K.layers.BatchNormalization()(conv_layer3)

    output = K.layers.add([conv_layer3, A_prev])
    return K.layers.Activation('relu')(output)
