#!/usr/bin/env python3
"""
Calculates the mean and covariance of a data set
"""
import numpy as np

def mean_cov(X):
    """
    Returns: mean, cov
    """
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    if (X.ndim) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    mean = X.mean(axis=0)
    mean = np.reshape(mean, (-1,3))
    X -= mean
    cov = np.dot(X.T, X) / len(X)
    return mean, cov
