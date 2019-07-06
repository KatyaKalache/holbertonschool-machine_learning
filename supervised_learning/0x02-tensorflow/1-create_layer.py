#!/usr/bin/env python3
"""
Layers
"""
import tensorflow as tf


def create_layer(prev, n_prev, n, activation):
    """
    Returns: the tensor output of the layer
    """
    
    with tf.name_scope('layer'):
        layer = tf.layers.Dense(units=n,
                                activation=activation,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"),
                                name='layer')
        return layer(prev)
