#!/usr/bin/env python3
import numpy as np
"""
Returns different operation to be performed on np arrays
"""


def np_elementwise(mat1, mat2):
    """
    Defines operations
    """
    add = np.add(mat1, mat2)
    sub = np.subtract(mat1, mat2)
    div = np.divide(mat1, mat2)
    mul = np.multiply(mat1, mat2)
    return(add, sub, mul, div)
