#!/usr/bin/env python3
"""
Trains a model using mini-batch gradient descent
"""


def train_model(network, data, labels, batch_size, epochs, verbose=True):
    """
    Returns: None
    """
    network.fit(data,
                labels,
                batch_size,
                epochs,
                verbose)
    
