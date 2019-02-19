#!/usr/bin/env python3

import numpy as np
"""
Concatenates numpy arrays at a specific axis
"""


def np_cat(mat1, mat2, axis=0):
    """
    Performing concatenation using numpy
    """
    res = np.concatenate((mat1, mat2), axis)
    return (res)
