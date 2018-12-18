#!/usr/bin/env python3
"""
concatenates two matrices along a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ Adding elements to either each element in the array
        Or to the whole array based on axis value
    """
    res = []
    if axis == 1:
        for i in mat1:
            for j in mat2:
                res.append(i+[j[0]])
                mat2.remove(j)

    if axis == 0:
        res.append(mat1 + mat2)
    return (res)
