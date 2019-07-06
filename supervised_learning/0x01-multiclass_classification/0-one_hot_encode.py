#!/usr/bin/env python3
"""
Converts a numeric label vector into a one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Returns one-hot encoding
    """
    return (np.eye(classes)[Y]).T
