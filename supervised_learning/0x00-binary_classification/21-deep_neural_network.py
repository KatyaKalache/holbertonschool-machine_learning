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
        Calculates the cost of the model using logistic regression
        """
        cost = (-Y * np.log(A) - (1 - Y) * np.log(1 - A)).mean()
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        """
        self.forward_prop(X)
        A = self.__cache["A{}".format(self.__L)] 
        cost = self.cost(Y, A)
        pred_labels = np.where(A >= 0.5, 1, 0)
        return pred_labels, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        """
        forward propagation
        """
        self.forward_prop(self.cache["A0"])
        """
        back propagation
        """
        m = Y[0].shape
        cache["dz{}".format(self.__L)] = cache["A{}".format(self.__L)] - Y
        cache["dw{}".format(self.__L)] = cache["dz{}".format(self.__L)] @ cache["A{}".format(self.__L - 1)].T / m
        cache["db{}".format(self.__L)] = np.sum(cache["dz{}".format(self.__L)], axis=1, keepdims=True) / m
        self.__L = self.__L + 1
        for i in range(1, self.__L - 1):
            cache["dz{}".format(self.__L - i - 1)] = self.__weights["W{}".format(self.__L - i)].T @ cache["dz{}".format(self.__L - i)] * cache["A{}".format(self.__L - i - 1)]
            cache["dw{}".format(self.__L - i - 1)] = cache["dz{}".format(self.__L - i - 1)] @ (cache["A{}".format(self.__L - i - 2)]).T / m
            cache["db{}".format(self.__L - i - 1)] = np.sum(cache["dz{}".format(self.__L - i - 1)], axis=1, keepdims=True) / m
            self.__weights["b{}".format(self.__L - i)] = self.__weights["b{}".format(self.__L - i)] - (alpha * cache["db{}".format(self.__L - i)])
            self.__weights["W{}".format(self.__L - i)] = self.__weights["W{}".format(self.__L - i)] - (alpha * cache["dw{}".format(self.__L - i)])
        
