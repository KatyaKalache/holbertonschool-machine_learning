#!/bin/usr/env python3
"""
calculates the shape of a matrix
"""


def matrix_shape(matrix):
    """
    Getting hight and width of the matrix
    Return new array with h and w
    """
    res = []
    width = 0
    height = 0
    for i in matrix:
        width = len(i)
        if len(i) > 2:
            for j in i:
                height = len(j)
    res.append(len(matrix))
    res.append(width)
    if (height > 0):
        res.append(height)
    return (res)
