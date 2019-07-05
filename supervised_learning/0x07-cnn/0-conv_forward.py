#!/usr/bin/env python3
"""
Performs forward propagation over a convolutional layer of a neural network
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Returns: the output of the convolutional layer
    """
    if padding == "valid":
        output_height = A_prev.shape[1] - W.shape[0] + 1
        output_width = A_prev.shape[2] - W.shape[0] + 1
        convolved_images = np.zeros((1, output_height, output_width, W.shape[3]))
        
    if padding == "same":
        output_height = A_prev.shape[1] // stride[0]
        output_width = A_prev.shape[2] // stride[1]
        convolved_images = np.zeros((1, output_height, output_width, W.shape[3]))

    kernel_height = W.shape[0]
    kernel_width = W.shape[1]
    for image in range(1):
        for fil in range(W.shape[3]):
            for i in range(convolved_images.shape[1] * convolved_images.shape[2]):
                convolved_images[image][i//convolved_images.shape[1],i%convolved_images.shape[2],fil] = activation(np.sum(A_prev[image][i//convolved_images.shape[1]*stride[0]:i//convolved_images.shape[1]*stride[1]+kernel_height, i%convolved_images.shape[2]*stride[0]:i%convolved_images.shape[2]*stride[1]+kernel_width, :] * W[:,:,:,fil]))+b[:,:,:,fil]
    return convolved_images
                

