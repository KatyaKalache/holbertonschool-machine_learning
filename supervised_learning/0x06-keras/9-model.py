#!/usr/bin/env python3
"""
1st function: saves an entire model
2nd function: loads an entire model
"""
import tensorflow.keras as K

def save_model(network, filename):
    """
    Returns: None
    """
    network.save(filename)
    

def load_model(filename):
    """
    Returns: the loaded model
    """
    loaded_model = K.models.load_model(filename)
    return loaded_model
    
    
