#!/usr/bin/env python3
"""
Determines if you should stop gradient descent early
"""


def early_stopping(cost, prev_cost, tolerance, patience, count):
    """
    Returns: a boolean of whether the network should be stopped early 
    followed by the updated patience count
    """
    
