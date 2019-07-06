#!/usr/bin/env python3
"""
Calculates the F1 score of a confusion matrix
"""
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Returns: an array containing the F1 score of each class
    f1 = 2 * (precision*recall/precision+recall)
    """
    i, l = 0, len(confusion)
    f1 = []
    while i < l:
        f1.append((2 * (sensitivity(confusion)[i] * precision(confusion)[i] /
                      (precision(confusion)[i] + sensitivity(confusion)[i]))))
        i += 1
    return f1
    
    
