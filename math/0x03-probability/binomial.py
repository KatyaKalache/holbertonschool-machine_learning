#!/usr/bin/env python3
"""
Create a class Binomial that represents a binomial distribution
"""
import numpy as np

class Binomial:
    """
    Sets the instance attributes n and p
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Class constructor
        """
        if data is not None:
            n = int(len(data) / 2)
            q = 1 - p
            for x in data:
                p += np.power(p,x)*np.power(q,n-x)
        self.n = n
        self.p = p
