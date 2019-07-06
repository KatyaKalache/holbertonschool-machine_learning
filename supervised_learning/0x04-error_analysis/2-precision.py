#!/usr/bin/env python3
"""
Calculates the precision for each class in a confusion matrix
"""


def precision(confusion):
    """
    Returns: array containing the precision of each class
    """
    column_sum = confusion.sum(axis=0)
    l, i = len(confusion), 0
    precision = []
    while i < l:
        for row in confusion:
            precision.append(row[i] / column_sum[i])
            i += 1
    return precision
