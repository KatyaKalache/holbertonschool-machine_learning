#!/usr/bin/env python3
"""
adds two matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    Checking length of matricies and its elements
    Appending one to another
    Return common array
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    res = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        res.append(row)
    return (res)
