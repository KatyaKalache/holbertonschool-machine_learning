#!/usr/bin/env python3
"""
Performs a valid convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Returns: a numpy.ndarray containing the convolved images
    """
    num_images = images.shape[0]
    output_height = images.shape[1] - kernel.shape[0] + 1
    output_width = images.shape[1] - kernel.shape[0] + 1
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    convolved_images = np.zeros((images.shape[0], output_height, output_width))

    for image in range(num_images):
        print (image)
        for i in range(convolved_images.shape[1]):
            for j in range(convolved_images.shape[2]):
                for fi in range(kernel_height):
                    for fj in range(kernel_width):
                        convolved_images[image][i][j] = np.matmul(images[image][i+fi][j+fj] * kernel[fi][fj])
    return convolved_images
