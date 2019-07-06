#!/usr/bin/env python3
"""
Tests a neural network
"""
import tensorflow.keras as K

def test_model(network, data, labels, verbose=True):
    """
    Returns: the loss and accuracy of the model with the testing data
    """
    return network.evaluate(data,
                            labels,
                            verbose=verbose)
    
