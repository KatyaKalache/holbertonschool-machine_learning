#!/usr/bin/env python3
"""
Makes a prediction using a neural network
"""


def predict(network, data, verbose=False):
    """
    Returns: the prediction for the data
    """
    return network.predict(data, verbose=verbose)


