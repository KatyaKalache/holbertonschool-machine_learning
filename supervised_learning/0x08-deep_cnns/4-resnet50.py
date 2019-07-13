#!/usr/bin/env python3
"""
Builds the ResNet-50 architecture
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Returns: the keras model
    """
    Input = K.layers.Input(shape=(224,224,3))
    conv1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer='he_normal', padding='same')(Input)
    batch_norm1 = K.layers.BatchNormalization()(conv1)
    activ1 = K.layers.Activation('relu')(batch_norm1)
    max_pool1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2))(activ1)

    conv2 = projection_block(max_pool1, [64, 64, 256])
    conv2= identity_block(conv2, [64, 64, 256])
    conv2= identity_block(conv2, [64, 64, 256])

    conv3 = projection_block(conv2, [128, 128, 512])
    conv3 = identity_block(conv3, [128, 128, 512])
    conv3 = identity_block(conv3, [128, 128, 512])
    conv3 = identity_block(conv3, [128, 128, 512])
        

    conv4 =  projection_block(conv3, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])

    conv5 = projection_block(conv4, [512, 512, 2048])
    conv5 = identity_block(conv5, [512, 512, 2048])
    conv5 = identity_block(conv5, [512, 512, 2048])

    avg_pool = K.layers.GlobalAveragePooling2D()(conv5)
    fully_connected = K.layers.Dense(1000)(avg_pool)
    activ = K.layers.Activation('softmax')(fully_connected)
    
    return K.models.Model(inputs=Input, outputs=activ)
