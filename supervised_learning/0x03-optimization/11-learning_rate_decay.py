#!/usr/bin/env python3
"""
Updates the learning rate using inverse time decay in numpy
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Returns: the updated value for alpha
    """
    alpha = 1 / (1+decay_rate * global_step) * alpha
    return alpha
