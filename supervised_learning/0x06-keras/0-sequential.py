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
    model = K.Sequential()
    num_layers = len(layers)
    kernel_regularizer=K.regularizers.l2(lambtha)
    for i in range(num_layers):
        model.add(K.layers.Dense(layers[i],
                                 kernel_regularizer=kernel_regularizer,
                                 activation=activations[i],
                                 input_dim=nx))
        if i < num_layers-1:
            model.add(K.layers.Dropout(keep_prob))
    return model
