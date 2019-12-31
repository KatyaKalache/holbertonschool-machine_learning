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
        self.load_model()
        self.model = model

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its values are between 0 and 1
        and its largest side is 512 pixels
        Returns the scaled image
        """
        height = image.shape[0]
        width = image.shape[1]
        new_width = 512
        new_height = int(new_width * height / width)
        image_scaled = tf.image.resize_images(
            image, (new_height, new_width),
            method=tf.image.ResizeMethod.BICUBIC)
        image_scaled = tf.reshape(image_scaled, (1, new_height, new_width, 3))
        norm_img = (image_scaled - np.min(image_scaled))/np.ptp(image_scaled)
        return norm_img

    def load_model(self):
        """
        Creates the model used to calculate cost
        """
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False)
        avg = tf.keras.Sequential()
        for layer in vgg.layers:
            layer.trainable = False
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer = tf.keras.layers.AveragePooling2D(name=layer.name)
                avg.add(layer)
            avg.add(layer)
        style_outputs = [avg.get_layer(layer).get_output_at(1)
                         for layer in self.style_layers]
        content_outputs = [avg.get_layer(layer).get_output_at(1)
                           for layer in self.content_layer]
        model_outputs = style_outputs + content_outputs
        global model
        model = tf.keras.models.Model(avg.layers[0].input, model_outputs)
