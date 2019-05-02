#!/usr/bin/env python3
"""
Class Neuron definning a single neuron
"""
import numpy as np


class Neuron:
    """
    Neuron class
    """
    def __init__(self, nx):
        """
        Defines a single neuron performing binary classification
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.W = np.random.normal(size=nx)
        self.b = 0
        self.A = 0
