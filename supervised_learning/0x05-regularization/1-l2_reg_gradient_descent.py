#!/usr/bin/env python3
"""
Updates the weights of a neural network with L2 regularization using gradient descent
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    The weights of the network should be updated in place
    """
    m = len(Y[0])
    cache["dz{}".format(L)] = cache["A{}".format(L)] - Y
    cache["dw{}".format(L)] = cache["dz{}".format(L)] @ cache["A{}".format(L - 1)].T / m + lambtha
    cache["db{}".format(L)] = np.sum(cache["dz{}".format(L)], axis=1, keepdims=True) / m
    L = L + 1
    for i in range(1, L - 1):
        cache["dz{}".format(L - i - 1)] = weights["W{}".format(L - i)].T @ cache["dz{}".format(L - i)] * cache["A{}".format(L - i - 1)]
        cache["dw{}".format(L - i - 1)] = cache["dz{}".format(L - i - 1)] @ (cache["A{}".format(L - i - 2)]).T / m + lambtha / m * weights["W{}".format(L-1-i)]
        cache["db{}".format(L - i - 1)] = np.sum(cache["dz{}".format(L - i - 1)], axis=1, keepdims=True) / m
        weights["b{}".format(L - i)] = weights["b{}".format(L - i)] - (alpha * cache["db{}".format(L - i)])
        weights["W{}".format(L - i)] = weights["W{}".format(L - i)] - (alpha * cache["dw{}".format(L - i)])

        
        
