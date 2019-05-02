#!/usr/bin/env python3
"""
Class DeepNeuralNetwork
Defines a deep multiple layers neural network
"""
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if min(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {"W1": np.random.randn(layers[0], nx), "b1": np.zeros((layers[0], 1))}

    @property
    def L(self):
        """
        Number of layers of NN
        """
        return self.__L

    @property
    def cache(self):
        """
        Intermediary values of the NN
        """
        return self.__cache

    @property
    def weights(self):
        """
        Weights and biases dict
        """
        return self.__weights
        
        for i in range(1, self.__L):
            self.__weights[("W{}".format(i+1))] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2/layers[i-1])
            self.__weights[("b{}".format(i+1))] = np.zeros((layers[i], 1))
            

