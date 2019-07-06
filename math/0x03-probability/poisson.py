#!/usr/bin/env python3
"""
Creates a class Poisson that represents a poisson distribution
"""
import numpy as np


class Poisson:
    """
    Class contructor
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Sets the instance attribute lambtha
        """
        e = 2.7182818285
        if data is not None:
            if  not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)
        else:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        k is the number of "successes"
        """
        e = 2.7182818285
        if not isinstance(k, int):
            k = int(k)
        pmf = np.power(e, -(self.lambtha)) * np.power(self.lambtha, k) / np.math.factorial(k)
        return pmf

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        """
        cdf = 0
        e = 2.7182818285
        if not isinstance(k, int):
            k =int(k)
        for i in range(k+1):
            cdf += np.power(self.lambtha, i) * np.power(e, -(self.lambtha))  / np.math.factorial(i)
        return cdf
