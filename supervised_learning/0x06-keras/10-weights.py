#!/usr/bin/env python3
"""
1st function: saves a model’s weights
2nd function: loads a model’s weights
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Returns: None
    """
    network.save_weights(filename, save_format=save_format)
    

def load_weights(network, filename):
    """
    Returns: None
    """
    network.load_weights(filename)
    
