#!/usr/bin/env python3
"""
Converts a label vector into a one-hot matrix
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Returns: the one-hot matrix
    """
    one_hot = K.utils.to_categorical(labels)
    return one_hot
