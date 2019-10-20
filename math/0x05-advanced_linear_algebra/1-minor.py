#!/usr/bin/env python3
"""
Calculates the minor matrix of a matrix
"""


def delcolumn(mat_copy, i):
    """
    Removes column
    """
    return [row[:i] + row[i+1:] for row in mat_copy]


def delrow(mat_copy, j):
    """
    Removes row
    """
    return mat_copy[:j] + mat_copy[j+1:]


def minor(matrix):
    """
    Returns: the minor matrix of matrix
    """
    copy_mat = matrix
    leng = len(matrix)
    if matrix == []:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]) or leng == 0 or len(matrix[0]) == 0:
        raise ValueError("Matrix must be a non-empty square matrix")
    if len(matrix[0]) == 1:
        return [[1]]
    if leng == 2:
        minor = [[0, 0], [0, 0]]
    if leng == 3:
        minor = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for row in range(leng):
        for col in range(leng):
            mat_rm_row = delrow(copy_mat, row)
            mat_rm_col = delcolumn(mat_rm_row, col)
            if leng == 3:
                minor[row][col] = mat_rm_col[0][0]*mat_rm_col[1][1]
                - mat_rm_col[0][1]*mat_rm_col[1][0]
            if leng == 2:
                minor[row][col] = mat_rm_col[0][0]
    return minor
