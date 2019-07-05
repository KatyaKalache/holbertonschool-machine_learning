#!/usr/bin/env python3
"""
Performs forward propagation over a pooling layer of a neural network
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Returns: the output of the pooling layer
    """
    pooled_height = (A_prev.shape[1] - kernel_shape[0])//stride[0]+1
    pooled_width = (A_prev.shape[2]- kernel_shape[1])//stride[1]+1
    pooled_depth = A_prev.shape[3]
    pooled_res = np.zeros((1, pooled_height, pooled_width, pooled_depth))

    for depth in range(pooled_depth):
        for i in range(pooled_height * pooled_width):
            if mode == "max":
                pooled_res[:1, i//pooled_res.shape[1], i%pooled_res.shape[1],depth] = np.max(A_prev[:1, i//pooled_res.shape[1]*stride[0]:i//pooled_res.shape[1]*stride[1]+kernel_shape[0], i%pooled_res.shape[2]*stride[0]:i%pooled_res.shape[2]*stride[1]+kernel_shape[1], :])
            if mode == "avg":
                pooled_res[:1, i//pooled_res.shape[1], i%pooled_res.shape[1],depth] = np.average(A_prev[:1, i//pooled_res.shape[1]*stride[0]:i//pooled_res.shape[1]*stride[1]+kernel_shape[0], i%pooled_res.shape[2]*stride[0]:i%pooled_res.shape[2]*stride[1]+kernel_shape[1], :])
    return pooled_res

