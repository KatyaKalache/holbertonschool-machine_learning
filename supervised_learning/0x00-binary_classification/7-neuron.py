#!/usr/bin/env python3
"""
Class Neuron that defines a single neuron
Evaluates the neuron’s predictions
"""
import numpy as np
import matplotlib.pyplot as plt

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

        self.__W = np.random.normal(size=(nx, 1))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Weights
        """
        return self.__W

    @property
    def b(self):
        """
        Bias
        """
        return self.__b

    @property
    def A(self):
        """
        Actvation
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        """
        z = np.dot(self.__W.T, X)+self.__b
        self.__A = 1/(1+np.exp(-z))
        return self.__A
 
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
        cost = self.cost(Y, self.A)
        pred_labels = np.where(self.A >= 0.5, 1, 0)
        return pred_labels, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        n = len(X[1])
        dz = A - Y
        # calculate partial derivative with respect to weigth
        # logistic regression
        dw = X @ dz.T / n
        # calculate partial derivative with respect to bias
        # logistic regression
        db = np.sum(dz.T) / n
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.5, verbose=True, graph=True, step=100):
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")

        if alpha < 0:
            raise ValueError("alpha must be positive")

        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A, alpha)
            if i % step == 0:
                cost = self.cost(Y, self.A)
                if verbose is True:
                    print ("Cost after {:d} iterations: {:f}".format(i, cost))
                    
                if graph is True:
                    x = np.arange(0, iterations, step)
                    plt.plot(x)
                    plt.show()
        pred_labels = self.evaluate(X, Y)
            
        return pred_labels
