#!/usr/bin/env python3
"""
Create a class Normal that represents a normal distribution
"""
import numpy as np


class Normal:
    """
    Class Normal that represents a normal distribution
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class constructor  
        """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = 0
            for i in range(len(data)-1):
                self.stddev += (np.power((data[i] - self.mean), 2)) / len(data)
            self.stddev = np.sqrt(self.stddev)
        else:
            if stddev < 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        Returns: z-score of x
        """
        z_score = (x - self.mean) / self.stddev
        return z_score

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        Returns: x-value of z
        """
        x_value = z * self.stddev + self.mean
        return x_value
        

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        Returns: the PDF value for x
        """
        π = 3.1415926536
        e = 2.7182818285
        power_for_e = np.power(-(x-self.mean), 2) / (2*np.power(self.stddev,2))
        e_to_power = np.power(e, power_for_e)
        pdf = 1/(self.stddev*(np.sqrt(2*π)) * e_to_power)
        return pdf

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        Returns cdf value
        """
        π = 3.1415926536
        e = 2.7182818285
        
        return cdf
