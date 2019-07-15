#!/usr/bin/env python3
"""
Builds a dense block
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Returns: The concatenated output of each layer 
    within the Dense Block and the number of filters 
    within the concatenated outputs, respectively
    """
    
    conv_list = [X]
    for i in range(layers):
        batch_norm = K.layers.BatchNormalization()(X)
        activ = K.layers.Activation('relu')(batch_norm)
        conv = K.layers.Conv2D(growth_rate*4, (1,1), kernel_initializer='he_normal', padding='same')(activ)
        batch_norm1 = K.layers.BatchNormalization()(conv)
        activ1 = K.layers.Activation('relu')(batch_norm1)
        conv1 = K.layers.Conv2D(growth_rate, (3,3), kernel_initializer='he_normal', padding='same')(activ1)
        conv_list.append(conv1)
        X = K.layers.concatenate([X, conv1])
        nb_filters += growth_rate 
    return conv_list, nb_filters
