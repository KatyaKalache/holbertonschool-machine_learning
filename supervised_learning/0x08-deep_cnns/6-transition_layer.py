#!/usr/bin/env python3
"""
Builds a transition layer 
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Returns: The output of the transition layer and 
    the number of filters within the output, respectively
    """
    filter_size = X.shape[1]
    batch_norm = K.layers.BatchNormalization()(X)
    activ = K.layers.Activation('relu')(batch_norm)
    conv = K.layers.Conv2D(int(nb_filters*compression), (1,1), kernel_initializer='he_normal')(activ)
    avg_pool = K.layers.AveragePooling2D(2, strides=(2,2))(conv)

    return avg_pool, avg_pool.shape[3]
