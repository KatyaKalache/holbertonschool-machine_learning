#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization
"""
import numpy as np
from numpy.linalg import norm


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Returns: the cost of the network accounting for L2 regularization
    """
    weights_to_power = {}
    weights_sum = 0
    for k, v in weights.items():
        weights_to_power[k] = np.sqrt(np.sum(np.power(v, 2)))
    for v in weights_to_power.values():
        weights_sum += v
    cost_function = cost + lambtha / (2*m) * weights_sum
    return cost_function
