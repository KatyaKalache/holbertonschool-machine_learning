#!/usr/bin/env python3
import numpy as np
"""
Returns a sum of power 2 digits
"""


def summation_i_squared(n):
    """
    Calculating the sum of power to 2
    """
    if isinstance(n, (float, int)) and n > 0:
        arr = np.power(np.arange(1, n+1, 1), 2)
        res = np.sum(arr)
        return (int(res))
    else:
        return None
