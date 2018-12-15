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

    res.append(len(matrix))
    res.append(len(matrix[0]))
    if len(matrix[0]) > 2:
        res.append(len(matrix[0][0]))
    return (res)
