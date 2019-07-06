#!/usr/bin/env python3
"""
Shuffles the data points in two matrices the same way
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Returns: the shuffled X and Y matrices
    """
    X_shuffled = np.random.permutation(X)
    Y_shuffled = np.random.permutation(Y)
    return X_shuffled, Y_shuffled
