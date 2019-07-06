#!/usr/bin/env python3
"""
Creates the training operation for the network
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Returns: an operation that trains the network 
    using gradient descent
    """
    train_op = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return train_op
