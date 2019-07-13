#!/usr/bin/env python3
"""
Builds a projection block
"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Returns: the activated output of the projection block
    """
    conv_2d = K.layers.Conv2D(filters[0], (1,1), strides=s, kernel_initializer='he_normal')(A_prev)
    batch_norm = K.layers.BatchNormalization()(conv_2d)
    activ = K.layers.Activation('relu')(batch_norm)
    
    conv2d_1 =  K.layers.Conv2D(filters[1], (3,3), padding='same', kernel_initializer='he_normal')(activ)
    batch_norm_1 = K.layers.BatchNormalization()(conv2d_1)
    activ_1 = K.layers.Activation('relu')(batch_norm_1)
    
    conv2d_2 =  K.layers.Conv2D(filters[2], (1,1), kernel_initializer='he_normal')(activ_1)
    batch_norm_2 = K.layers.BatchNormalization()(conv2d_2)
    activ_3 = K.layers.Activation('relu')(batch_norm_2)
    
    shortcut = K.layers.Conv2D(filters[2], (1, 1), strides=s, kernel_initializer='he_normal')(A_prev)
    shortcut_batch_norm = K.layers.BatchNormalization()(shortcut)
    activ_shortcut =  K.layers.Activation('relu')(shortcut_batch_norm)
    
    output = K.layers.add([activ_3, activ_shortcut])

    return output
