#!/usr/bin/env python3
"""
Normalizes (standardizes) a matrix
"""
import numpy as np
import tensorflow as tf


def normalize(X, m, v, epsilon):
    """
    Returns: The normalized X matrix
    """
    norm = (X - m) / np.sqrt(v + epsilon) ** 2
    return norm
