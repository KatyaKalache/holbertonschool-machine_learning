#!/usr/bin/env python3
"""
Creates the forward propagation graph for the neural network
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, nx, layer_sizes=[], activations=[]):
    """
    Returns: the prediction of the network in tensor form
    """
    n = x.shape[1]
    with tf.name_scope('layer'):
        layer = create_layer(x, n, layer_sizes[1], activations[0])
    layer_len = len(layer_sizes)
    
    for i in range(1, layer_len):
        layer = create_layer(layer[1:], layer_len, layer_sizes[i], activations[i])
    return layer
