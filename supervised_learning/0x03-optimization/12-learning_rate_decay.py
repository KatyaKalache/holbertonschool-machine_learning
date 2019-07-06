#!/usr/bin/env python3
"""
Creates a learning rate decay operation in tensorflow 
using inverse time decay
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Returns: the learning rate decay operation
    """
    alpha = tf.train.inverse_time_decay(learning_rate=alpha,
                                        global_step=global_step,
                                        decay_steps=decay_step,
                                        decay_rate=decay_rate,
                                        staircase=True)

    return alpha
