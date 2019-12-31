#!/usr/bin/env python3
"""
Create a class NST that performs tasks for neural style transfer
"""
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()


class NST:
    """
    Class NST
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e1, beta=1e3):
        """
        Class constructor
        """
        if type(style_image) is not np.ndarray or \
           len(style_image.shape) != 3 or \
           style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if type(content_image) is not np.ndarray or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = float(10000)
        self.beta = 1

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its values are between 0 and 1
        and its largest side is 512 pixels
        Returns the scaled image
        """
        height = image.shape[0]
        width = image.shape[1]
        new_width = int(width / max(image.shape)*512)
        new_height = int(height / max(image.shape)*512)
        image_scaled = tf.image.resize_images(
            image, (new_height, new_width),
            method=tf.image.ResizeMethod.BICUBIC)
        image_scaled = tf.reshape(image_scaled, (1, new_height, new_width, 3))
        norm_img = image_scaled / np.max(image_scaled)
        return abs(norm_img)
