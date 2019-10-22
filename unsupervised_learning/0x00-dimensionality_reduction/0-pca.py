#!/usr/bin/env python3
"""
 Performs PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """
    Returns: the weights matrix, W,
    that maintains var fraction of X‘s original variance
    """
    M = np.mean(X, axis=0)
    C = X - M
    U, S, V = np.linalg.svd(C, full_matrices=False)
    return V[:3].T
