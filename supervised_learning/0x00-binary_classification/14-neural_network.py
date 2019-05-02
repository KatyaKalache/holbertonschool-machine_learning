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

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.A2)
        pred_labels = np.where(self.A2 >= 0.5, 1, 0)
        return pred_labels, cost


    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        n = len(X[0])
        """
        forward propagation
        """
        z1 = self.__W1 @ X + self.__b1
        z2 = self.__W2 @ A1 + self.__b2
                                                                            
        """
        back propagation
        """
        dz2 = A2 - Y
        dw2 = dz2 @ A1.T / n
        db2 = np.sum(dz2, axis=1, keepdims=True) / n
        dz1 = self.__W2.T @ dz2 * A1
        print ("W2", self.__W2.T.shape)
        print ("dz2", dz2.shape)
        print ("A1", A1.shape)
        dw1 =  dz1 @ X.T / n
        db1 = np.sum(dz1, axis=1, keepdims=True) / n
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)
                
    def train(self, X, Y, iterations=5000, alpha=0.05):
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)
        pred_labels = self.evaluate(X, Y)
        return pred_labels
