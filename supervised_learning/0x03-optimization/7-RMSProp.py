#!/usr/bin/env python3
"""
Updates a variable using the RMSProp optimization algorithm
"""
import numpy as np

def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Returns: the updated variable and the new moment, respectively
    """
    S =  beta2 * s + (1 - beta2) * np.power(grad, 2)
    W = var - alpha * grad/(np.sqrt(S)+epsilon)
    return W, S
