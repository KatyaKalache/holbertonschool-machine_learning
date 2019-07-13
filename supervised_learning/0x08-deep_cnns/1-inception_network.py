#!/usr/bin/env python3
"""
Builds the inception network
Based on GoogLeNet architecture
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Returns: the keras model
    """
    Input = K.layers.Input(shape=(224,224,3))

    conv_layer1 = K.layers.Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu')(Input)
    layer1_max_pooling = K.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv_layer1)

    conv_layer2_reduce = K.layers.Conv2D(64, (1,1), padding='same', activation='relu')(layer1_max_pooling)
    conv_layer2 =  K.layers.Conv2D(192, (3,3), strides=(1,1), padding='same', activation='relu')(conv_layer2_reduce)
    layer2_max_pooling = K.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', data_format='channels_last')(conv_layer2)

    inception_3a = inception_block(layer2_max_pooling,(64,96,128,16,32,32))

    inception_3b = inception_block(inception_3a,(128,128,192,32,96,64))

    max_pooling_3 = K.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(inception_3b)

    inception_4a = inception_block(max_pooling_3,(192,96,208,16,48,64))

    inception_4b = inception_block(inception_4a,(160,112,224,24,64,64))

    inception_4c = inception_block(inception_4b,(128,128,256,24,64,64))

    inception_4d = inception_block(inception_4c,(112,144,288,32,64,64))

    inception_4e = inception_block(inception_4d,(256,160,320,32,128,128))
    
    max_pooling_4 = K.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', data_format='channels_last')(inception_4e)

    inception_5a = inception_block(max_pooling_4,(256,160,320,32,128,128))

    inception_5b = inception_block(inception_5a,(384,192,384,48,128,128))

    avg_pooling = K.layers.AveragePooling2D(pool_size=(7,7), strides=(1,1), padding='valid')(inception_5b)

    drop_40 = K.layers.Dropout(0.4)(avg_pooling)

    fully_connected = K.layers.Dense(1000)(drop_40)

    activation = K.layers.Activation(activation='softmax')(fully_connected)

    googlenet = K.models.Model(inputs=Input, outputs=activation)

    return googlenet
