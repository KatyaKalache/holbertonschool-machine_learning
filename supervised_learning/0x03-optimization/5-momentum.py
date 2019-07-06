#!/usr/bin/env python3
"""
Updates a variable using the gradient descent with momentum optimization algorithm
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Returns: the updated variable and the new moment, respectively
    """
    V = beta1 * v + (1 - beta1) * grad
    W = var - alpha * V

    return W, V
