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
        if len(mat1) == len(mat2):
            for i in mat1:
                for j in mat2:
                    res.append(i+[j[0]])
                    mat2.remove(j)
        else:
            return None
    else:
        for i in mat2:
            if len(i) == len(mat1[0]):
                res.append(mat1 + [i])
            else:
                return None
    return (res)
