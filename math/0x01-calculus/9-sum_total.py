#!/usr/bin/env python3
"""
Returns a sum of power 2 digits
"""


def summation_i_squared(n):
    """
    Calculating the sum of power to 2
    """
    if isinstance(n, (float, int)) and n > 0:
        res = n*(n+1)*(2*n+1)/6
        return (int(sum(res)))
    else:
        return None
