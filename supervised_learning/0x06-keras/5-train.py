#!/usr/bin/env python3
"""
Trains a model using mini-batch gradient descent
based on 4-train.py
"""


def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True):
    """
    Returns: Nothing
    """
    network.fit(data,
                labels,
                batch_size,
                epochs,
                validation_data=validation_data,
                verbose=verbose)
    
