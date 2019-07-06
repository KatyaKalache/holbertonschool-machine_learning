#!/usr/bin/env python3
"""
Updates a variable using the Adam optimization algorithm
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Returns: the updated variable, the new first moment, and the new second moment
    """
    V = beta1 * v + (1 - beta1) * grad
    S =  beta2 * s + (1 - beta2) * np.power(grad, 2)
    V_cor = V / (1-np.power(beta1, t))
    S_cor = S / (1-np.power(beta2, t))
    # update
    W =  var - alpha * V / (np.sqrt(S)+epsilon)

    return W,V,S
