#!/usr/bin/env python3
"""
Clculates the cost of a neural network with L2 regularization
"""
import tensorflow as tf

def l2_reg_cost(cost):
    """
    Returns: a tensor containing the cost of the network 
    accounting for L2 regularization
    """
    cost += tf.losses.get_regularization_losses()
    return cost
