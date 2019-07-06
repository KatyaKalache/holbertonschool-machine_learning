#!/usr/bin/env python3
"""
Creates a tensorflow layer that includes L2 regularization
"""
import tensorflow as tf

def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Returns: the output of the new layer
    """
    regulizer = tf.contrib.layers.l2_regularizer(scale=lambtha)
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(kernel_initializer=initializer,
                            kernel_regularizer=regulizer,
                            units=n,
                            activation=activation)
    return layer(prev)
