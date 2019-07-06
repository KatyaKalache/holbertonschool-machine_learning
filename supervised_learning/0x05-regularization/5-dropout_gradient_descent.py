#!/usr/bin/env python3
"""
Updates the weights of a neural network with Dropout 
regularization using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Returns: list of updated weights
    """
    m = Y.shape[1]
    dz = {}
    dz["dz{}".format(L)] = cache["A{}".format(L)] - Y
    cache["dw{}".format(L)] = dz["dz{}".format(L)] @ cache["A{}".format(L-1)].T / m
    cache["db{}".format(L)] = np.sum(dz["dz{}".format(L)], axis=1, keepdims=True) / m

    L +=  1
    i = 1

    cache["dA{}".format(L-i-1)] = np.dot(weights["W{}".format(L-i)].T, dz["dz{}".format(L-i)])
    cache["dA{}".format(L-i-1)] = np.multiply(cache["D{}".format(L-i-1)], cache["dA{}".format(L-i-1)])
    cache["dA{}".format(L-i-1)] = cache["dA{}".format(L-i-1)]  / keep_prob    
    dz["dz{}".format(L-i-1)] = np.multiply(cache["dA{}".format(L-i-1)], np.int64(cache["A{}".format(L-i-1)] > 0))
    cache["dw{}".format(L-i-1)] = dz["dz{}".format(L-i-1)] @ cache["A{}".format(L-i-2)].T / m
    cache["db{}".format(L-i-1)] = np.sum(dz["dz{}".format(L-i-1)], axis=1, keepdims=True) / m
    weights["b{}".format(L-i)] = weights["b{}".format(L-i)] - (alpha * cache["db{}".format(L-i)])
    weights["W{}".format(L-i)] = weights["W{}".format(L-i)]- (alpha * cache["dw{}".format(L-i)])

    i = 2
    
    cache["dA{}".format(L-i-1)] = np.dot(weights["W{}".format(L-i)].T, dz["dz{}".format(L-i)])
    cache["dA{}".format(L-i-1)] = np.multiply(cache["D{}".format(L-i-1)], cache["dA{}".format(L-i-1)])
    cache["dA{}".format(L-i-1)] = cache["dA{}".format(L-i-1)]  / keep_prob
    dz["dz{}".format(L-i-1)] = np.multiply(cache["dA{}".format(L-i-1)], np.int64(cache["A{}".format(L-i-1)] > 0))
    cache["dw{}".format(L-i-1)] = dz["dz{}".format(L-i-1)] @ cache["A{}".format(L-i-2)].T / m
    cache["db{}".format(L-i-1)] = np.sum(dz["dz{}".format(L-i-1)], axis=1, keepdims=True) / m
    weights["b{}".format(L-i)] = weights["b{}".format(L-i)] - (alpha * cache["db{}".format(L-i)])
    weights["W{}".format(L-i)] = weights["W{}".format(L-i)]- (alpha * cache["dw{}".format(L-i)])
