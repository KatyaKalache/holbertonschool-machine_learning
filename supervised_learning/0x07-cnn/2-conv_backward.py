#!/usr/bin/env python3
"""
Performs back propagation over a convolutional layer of a neural network
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Returns: the partial derivatives with respect to the previous layer (dA_prev), 
    the kernels (dW), and the biases (db)
    """
    
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    if padding == "valid":
        output_height = A_prev.shape[1] - W.shape[0] + 1
        output_width = A_prev.shape[2] - W.shape[0] + 1
        output = np.zeros((1, output_height, output_width, W.shape[3]))
        dA_prev = np.zeros((1, output_height, output_width, W.shape[3]))


    if padding == "same":
        output_height = A_prev.shape[1] // stride[0]
        output_width = A_prev.shape[2] // stride[1]
        output = np.zeros((1, output_height, output_width, W.shape[3]))
        dA_prev = np.zeros((1, output_height, output_width, W.shape[3]))
    
    for image in range(1):
        for channel in range(dZ.shape[3]):
            for h in range(dZ.shape[1]):
                for w in range(dZ.shape[2]):
                    dA_prev[image, h*stride[0]:h*stride[0]+W.shape[0], w*stride[1]:w*stride[1]+W.shape[1], :] += W[:,:,:,channel] * dZ[image, h, w, channel]
                    dW[:,:,:, channel] += output[image, h*stride[0]:h*stride[0]+W.shape[0], w*stride[1]:w*stride[1]+W.shape[1],:] * dZ[image, h, w,channel]
                    db[:,:,:,channel] += dZ[image, h, w, channel]
    return dA_prev, dW, db

