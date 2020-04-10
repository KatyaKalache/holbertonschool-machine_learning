#!/usr/bin/env python3
"""
Builds the DenseNet-121 architecture
"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Returns: The output of the transition layer and 
    the number of filters within the output, respectively
    """
    Input = K.layers.Input(shape=(224,224,3))
    batch_norm1 = K.layers.BatchNormalization()(Input)
    activ1 = K.layers.Activation('relu')(batch_norm1)
    conv1 = K.layers.Conv2D(64, (7,7),  strides=(2, 2), kernel_initializer='he_normal', padding='same')(activ1)
    max_pool1 = K.layers.MaxPooling2D((3, 3), padding='same', strides=(2, 2))(conv1)
    
    dense_block1 = dense_block(max_pool1, 56, growth_rate, 6)
    
    for i in dense_block1[0]:
        transition_block1 = transition_layer(i, 1, compression)
    avg_pool = K.layers.GlobalAveragePooling2D()(transition_block1[0])
    fully_connected = K.layers.Dense(1000)(avg_pool)
    activ = K.layers.Activation('softmax')(fully_connected)
    return K.models.Model(inputs=Input, outputs=activ)

