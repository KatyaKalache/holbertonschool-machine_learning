#!/usr/bin/env python3
"""
Sets up Adam optimization for a keras model
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Returns: None
    """
    optimizer = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

