#!/usr/bin/env python3
"""
Calculates the weighted moving average of a data set
"""
import numpy as np

def moving_average(data, beta):
    """
    Returns: a list containing the moving averages of data
    """
    # beta - weight
    avg_list = []
    mov_avg = 0
    for i in range(len(data)):
        mov_avg = ((mov_avg * beta) + ((1 - beta) * data[i]))
        mov_avg_cor = mov_avg / (1 - beta ** (i + 1))
        avg_list.append(mov_avg_cor)
    
    return avg_list
        
        
