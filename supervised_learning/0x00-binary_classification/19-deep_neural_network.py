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
        for i in range(1, self.__L):
            self.__weights[("W{}".format(i+1))] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2/layers[i-1])
            self.__weights[("b{}".format(i+1))] = np.zeros((layers[i], 1))
            
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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the NN
        """
        self.__cache = {"A0": X}
        self.z = {}
        for i in range(1, self.__L + 1):
            self.z["z{}".format(i)] = (self.__weights["W{}".format(i)] @ self.__cache["A{}".format(i-1)]) + self.__weights["b{}".format(i)]
            self.__cache["A{}".format(i)] = 1/(1+np.exp(-(self.z["z{}".format(i)])))
        return self.__cache["A{}".format(i)], self.__cache

    def cost(self, Y, A):
        """
        Evaluates the neuron’s predictions
        """
        cost = (-Y * np.log(A) - (1 - Y) * np.log(1 - A)).mean()
        return cost
