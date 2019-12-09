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
    content_layer = ['block5_conv2']

    def __init__(self, style_image, content_image, alpha=1e1, beta=1e3):
        """
        Class constructor
        """
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = 10000
        self.beta = 1

    def scale_image(self, image):
        """
        Rescales an image such that its values are between 0 and 1
        and its largest side is 512 pixels
        Returns the scaled image
        """
        global image
        height = image.shape[0]
        width = image.shape[1]
        new_width = 512
        new_height = int(new_width * height / width)
        image = tf.image.resize_images(image, (new_height, new_width))
        image = np.resize(image, (1, new_height, new_width, 3))
        image = (image - np.min(image))/np.ptp(image)
        return image
