#!/usr/bin/env python3
"""
Normalizes an unactivated output of a neural network 
using batch normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Returns: the normalized Z matrix
    """
    m = Z.shape[0]
    batch_mean = np.sum(Z) / m
    batch_variance = np.sum((Z - batch_mean)**2)  / m
    z_norm = (Z - batch_mean) / np.sqrt(batch_variance+epsilon)
    Z_norm = gamma * z_norm + beta
    return Z_norm
