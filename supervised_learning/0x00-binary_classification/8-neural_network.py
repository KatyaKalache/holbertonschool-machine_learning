#!/usr/bin/env python3
"""
Class NeuralNetwork
Defines a neural network with one hidden
"""
import numpy as np

class NeuralNetwork:
    """
    Performing binary classification
    On 2 layers Neural Network
    """
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be a integer")
        if nodes < 1:
            raise ValueError("nodes must be positive")
        self.W1 = np.random.normal(size=(nx, 1))
        self.b1 = np.zeros(nodes)
        self.A1 = 0
        self.W2 = np.random.normal(size=(nodes, 1))
        self.b2 = 0
        self.A2 = 0
