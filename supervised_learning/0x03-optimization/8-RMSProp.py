#!/usr/bin/env python3
"""
creates the training operation for a neural network in tensorflow
using the RMSProp optimization algorithm
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Returns: the RMSProp optimization operation
    """
    train_op = tf.train.RMSPropOptimizer(learning_rate=alpha, momentum=beta2, epsilon=epsilon).minimize(loss)
    return train_op
