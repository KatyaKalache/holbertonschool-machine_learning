#!/usr/bin/env python3
"""
transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Transposing matriecies by flipping rows with columns
    Return transposed matrix as a new 2d array
    """
    res = [[matrix[j][i] for j in range(len(matrix))] for i in
           range(len(matrix[0]))]
    return (res)
