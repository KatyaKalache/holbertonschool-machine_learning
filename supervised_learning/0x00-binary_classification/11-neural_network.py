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
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0


    @property
    def W1(self):
        """
        Weights for hidden layer
        """
        return self.__W1

    @property
    def b1(self):
        """
        Bisas for hidden layer
        """
        return self.__b1

    @property
    def A1(self):
        """
        Activated output for hidden layer
        """
        return self.__A1

    @property
    def W2(self):
        """
        Weights for output neuron
        """
        return self.__W2

    @property
    def b2(self):
        """
        Bias for the output neuron
        """
        return self.__b2

    @property
    def A2(self):
        """
        Activated output for the output neuron
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1/(1+np.exp(-z1))
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1/(1+np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        cost = (-Y * np.log(A) - (1 - Y) * np.log(1 - A)).mean()
        return cost
