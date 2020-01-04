#!/usr/bin/env python3
"""
Create the class Multinormal that represents
a Multivariate Normal distribution
"""
import numpy as np
from scipy.stats import norm

class MultiNormal:
    """
    Class constructor
    """
    def __init__(self, data):
        """
        Set the public instance variables:
        mean - mean of data
        cov - covariance matrix data 
        """
        self.mean = np.mean(data, axis=1)
        mean_x = np.mean(data[0,:])
        mean_y = np.mean(data[1,:])
        mean_z = np.mean(data[2,:])
        mean_vector = np.array([[mean_x],[mean_y],[mean_z]])
        scatter_matrix = np.zeros((3,3))
        for i in range(data.shape[1]):
            scatter_matrix += (data[:,i].reshape(3,1) - mean_vector).dot((data[:,i].reshape(3,1) - mean_vector).T / data.shape[1])
        self.cov = scatter_matrix

    def pdf(self, x):
        """

        """
        mean = np.mean(x, axis=0)
        cov = np.cov(x.T)
        pdf = multivariate_normal.pdf(x, mean=mean, cov=cov)    
        return (pdf)
