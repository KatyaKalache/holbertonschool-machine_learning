#!/usr/bin/env python3
"""
Builds a neural network with the Keras library
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Returns: the keras model
    """
    # create model
    inputs = K.Input(shape=(nx,))
    num_layers = len(layers)
    kernel_regularizer=K.regularizers.l2(lambtha)
    layer = K.layers.Dense(kernel_regularizer=kernel_regularizer,
                           units=layers[0],
                           activation=activations[0],
                           input_dim=nx)(inputs)
    layer = K.layers.Dropout(keep_prob)(layer)    
    for i in range(1, num_layers):
        layer = K.layers.Dense(kernel_regularizer=kernel_regularizer,
                               units=layers[i],
                               activation=activations[i],
                               input_dim=nx)(layer)
        if i < (num_layers-1):
            layer = K.layers.Dropout(keep_prob)(layer)
    model = K.Model(inputs, layer)
    return model
