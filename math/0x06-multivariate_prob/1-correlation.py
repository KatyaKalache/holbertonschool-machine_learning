#!/usr/bin/env python3
"""
Calculates a correlation matrix
"""
import numpy as np


def correlation(C):
    """
    Returns a numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    cor = np.corrcoef(C)
    return cor
