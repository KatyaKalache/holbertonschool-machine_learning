#!/usr/bin/env python3
"""
Class Exponential that represents an exponential distribution
"""
import numpy as np


class Exponential:
    """
    Class constructor
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Sets the instance attribute lambtha
        """
        if data is not None:
            if  not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            self.lambtha = 1 / mean
        else:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period
        """
        e = 2.7182818285
        pdf = self.lambtha * np.power(e, (-(self.lambtha)*x)) 
        return pdf

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period
        """
        e = 2.7182818285
        cdf = 1 - np.power(e, (-(self.lambtha)*x))
        return cdf
