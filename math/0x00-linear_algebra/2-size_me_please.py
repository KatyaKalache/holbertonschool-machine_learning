#!/bin/usr/env python3
def matrix_shape(matrix):
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
    if (height):
        res.append(height)
    return (res)
