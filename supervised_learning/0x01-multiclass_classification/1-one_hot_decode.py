#!/usr/bin/env python3
"""
Converts a one-hot matrix into a vector of labels
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Returns array containing the numeric labels for each example
    """
    return np.argmax(one_hot, axis=0)
