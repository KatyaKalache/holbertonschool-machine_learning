#!/usr/bin/env python3
"""
Conducts forward propagation using Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Returns: a dictionary containing the outputs of each layer 
    and the dropout mask used on each layer 
    """
    cache = {'A0': X}
    z = {}
    for i in range(1, L+1):
        z["z{}".format(i)] = np.add(np.matmul(weights["W{}".format(i)], cache["A{}".format(i-1)]), weights["b{}".format(i)])
        cache["A{}".format(i)] = np.tanh(z["z{}".format(i)])
        # dropout implementation
        # initialize matrix D1 of A1 shape
        cache["D{}".format(i)] = np.random.rand(cache["A{}".format(i)].shape[0], cache["A{}".format(i)].shape[1])
        # convert entries of D1 to 0 or 1 
        cache["D{}".format(i)] = cache["D{}".format(i)] < keep_prob
        # shut down some neurons of A
        cache["A{}".format(i)] = np.multiply(cache["A{}".format(i)], cache["D{}".format(i)])
        # scale the value of neurons that haven't been shut down
        cache["A{}".format(i)] = cache["A{}".format(i)]/keep_prob
    return  cache
