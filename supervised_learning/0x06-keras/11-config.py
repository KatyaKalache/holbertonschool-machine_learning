#!/usr/bin/env python3
"""
1st fucntion: saves a model’s configuration in JSON format
2nd function: loads a model with a specific configuration
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Returns: None
    """
    json_model = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_model)
    
def load_config(filename):
    """
    Returns: the loaded model
    """
    with open(filename, 'r') as f:
        return K.models.model_from_json(f.read())

