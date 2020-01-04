#!/usr/bin/env python3
"""
Calculates the likelihood of obtaining this data
"""
from math import factorial
import numpy as np


def likelihood(x, n, P):
    """
    Returns: a 1D numpy.ndarray containing the likelihood 
    of obtaining the data, x and n, for each probability in P
    """
    if P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if 0 < P.any() > 1:
        raise ValueError("All values in P must be in the range [0, 1]")
    res = []
    for theta in P:
        res.append((factorial(n) /(factorial(x) * factorial(n - x)))* (theta ** x) * ((1 - theta) ** (n - x)))
    return res
