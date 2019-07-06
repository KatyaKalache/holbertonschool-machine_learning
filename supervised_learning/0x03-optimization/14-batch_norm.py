#!/usr/bin/env python3
"""
Creates a batch normalization layer for a neural network in tensorflow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Returns: a tensor of the activated output for the layer
    """
    batch_norm = tf.nn.batch_normalization()
