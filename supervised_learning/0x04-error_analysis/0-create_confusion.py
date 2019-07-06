#!/usr/bin/env python3
"""
Creates a confusion matrix
"""
import numpy as np

def create_confusion_matrix(labels, logits):
    """
    Returns: a confusion numpy.ndarray of shape (classes, classes) 
    with row indices representing the correct labels and 
    column indices representing the predicted labels
    """

    classes = labels.shape[1]
    conf_mat = np.zeros((classes, classes))
    for i, j in zip(labels, logits):
        conf_mat[np.argmax(i)][np.argmax(j)] += 1
        
    return conf_mat
