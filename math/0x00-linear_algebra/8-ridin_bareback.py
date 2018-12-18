#!/usr/bin/env python3
"""
performs matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
    Return a new matrix
    Multiplication of all elements of 2 matricies
    """
    if len(mat1[0]) != len(mat2):
        return None

    res = [[0 for i in range(len(mat2[0]))] for j in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for x in range(len(mat2)):
                res[i][j] += mat1[i][x] * mat2[x][j]
    return(res)
