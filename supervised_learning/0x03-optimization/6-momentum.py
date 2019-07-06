#!/usr/bin/env python3
"""
Creates the training operation for a neural network
in tensorflow using the gradient descent with momentum optimization algorithm
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Returns: the momentum optimization operation
    """
    train_op = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1).minimize(loss)
    return train_op

