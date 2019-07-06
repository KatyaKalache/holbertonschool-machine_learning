#!/usr/bin/env python3
"""
Creates a layer of a neural network using dropout
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Returns: the output of the new layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer)
    layer_with_dropout = tf.layers.Dropout(1-keep_prob)
    return layer_with_dropout(layer(prev))
