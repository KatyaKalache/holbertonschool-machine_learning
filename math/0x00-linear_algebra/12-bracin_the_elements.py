#!/usr/bin/env python3
"""
Returns different operation to be performed on np arrays
"""


def np_elementwise(mat1, mat2):
    """
    Defines operations
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    div = mat1 / mat2
    mul = mat1 * mat2
    return(add, sub, mul, div)
