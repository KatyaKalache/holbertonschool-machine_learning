#!/usr/bin/env python3
"""
Calculates the sensitivity for each class in a confusion matrix
"""
import numpy as np


def sensitivity(confusion):
    """
    Returns: a numpy.ndarray of shape (classes,) 
    containing the sensitivity of each class
    sensitivity = TP/actual_yes
    precision = TP/predicted_yes
    specifisity = TN/actual_no 
    """
    l = len(confusion)
    sensitivity = []
    i = 0
    while (i < l):
        for row in confusion:
            sensitivity.append(row[i] / sum(row))
            i += 1
    return sensitivity



