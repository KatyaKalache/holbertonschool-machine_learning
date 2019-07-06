#!/usr/bin/env python3
"""
Calculates the specificity for each class in a confusion matrix
"""


def specificity(confusion):
    """
    Returns: an array containing the specificity of each class
    specificity = TN / actual_no
    """
    l = len(confusion)
    i = 0
    column_sum = confusion.sum(axis=0)
    # TN = each_in_column - actual_yes
    while i < l:
        for row in confusion:
            TN = confusion[:, i] - row[i]
            
            i+= 1
