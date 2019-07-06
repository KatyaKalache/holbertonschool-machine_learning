#!/usr/bin/env python3
"""
Creates the training operation for a neural network in tensorflow
using the Adam optimization algorithm
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Returns: the Adam optimization operation
    """
    train_op = tf.train.AdamOptimizer(learning_rate=alpha, epsilon=epsilon, beta1=beta1, beta2=beta2).minimize(loss)
    return train_op
