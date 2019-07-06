#!/usr/bin/env python3
"""
Performs a valid convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Returns: a numpy.ndarray containing the convolved images
    """
    num_images = images.shape[0]
    output_height = images.shape[1] - kernel.shape[0] + 1
    output_width = images.shape[1] - kernel.shape[0] + 1
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    convolved_images = np.zeros((images.shape[0], output_height, output_width))

    for image in range(1):
        for i in range(convolved_images.shape[1] * convolved_images.shape[2]):
            image_slice = images[image][i//convolved_images.shape[1]:i//convolved_images.shape[2]+kernel_height, i%convolved_images.shape[1]:i%convolved_images.shape[2]+kernel_width]
            convolved_images[image][i//convolved_images.shape[1]][i%convolved_images.shape[2]] = np.sum(image_slice * kernel)
    print (convolved_images)
    return convolved_images
