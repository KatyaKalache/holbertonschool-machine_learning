#!/usr/bin/env python3
"""
Returns derivative polynomial array
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial
    """
    if poly == 0:
        return([0])
    res = []
    i = 0
    while i < len(poly):
        print(poly)
        res.append(poly[i]*i)
        i += 1
    return (res)
