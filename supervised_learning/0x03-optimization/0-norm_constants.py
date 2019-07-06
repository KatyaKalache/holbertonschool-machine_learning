#!/usr/bin/env python3
"""
Calculates the normalization (standardization) constants of a matrix
"""
import numpy as np
import statistics


def normalization_constants(X):
    """
    Returns: the mean and variance of each feature
    """
    """ mean of each column """
    mean = np.mean(X, axis=0)
    variance = np.mean((X-mean)**2, axis=0)
    
    return mean, variance

