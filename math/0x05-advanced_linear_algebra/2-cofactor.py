#!/usr/bin/env python3
"""
Calculates the cofactor matrix of a matrix
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

def cofactor(matrix):
    """
    Returns: the cofactor matrix of matrix
    """
    copy_mat = matrix
    leng = len(matrix)
    if matrix == []:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if len(matrix[0]) == 1:
        return [[1]]
    if leng == 2:
        minor = [[0,0],[0,0]]
    if leng == 3:
        minor = [[0,0,0], [0,0,0], [0,0,0]]
    for row in range(leng):
        for col in range(leng):
            mat_rm_row = delrow(copy_mat, row)
            mat_rm_col = delcolumn(mat_rm_row, col)
            if leng == 3:
                minor[row][col] = mat_rm_col[0][0]*mat_rm_col[1][1] - mat_rm_col[0][1]*mat_rm_col[1][0]
            if leng == 2:
                minor[row][col] = mat_rm_col[0][0]
    minor_1d = sum(minor, [])
    if len(minor) == 3:
        for i in range(len(minor)):
            if i % 2 != 0:
                minor_1d[i] = minor_1d[i] * -1
        cof_mat = [minor_1d[i:i+3] for i in range(0, len(minor_1d), 3)]
    if len(minor) == 2:
        minor_1d[1] = minor_1d[1] * -1
        minor_1d[2] = minor_1d[2] * -1
        cof_mat = [minor_1d[i:i+2] for i in range(0, len(minor_1d), 2)]
    return cof_mat
