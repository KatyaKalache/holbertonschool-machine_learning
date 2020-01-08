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

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor
        """
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or \
           style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or \
           content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.model = model

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its values are between 0 and 1
        and its largest side is 512 pixels
        Returns the scaled image
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or \
           image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        height = image.shape[0]
        width = image.shape[1]
        scale = 512 / max(height, width)
        new_shape = (int(scale * height), int(scale * width))
        image = np.expand_dims(image, axis=0)
        image_scaled = tf.clip_by_value(tf.image.resize_bicubic
                                        (image, new_shape) / 255.0, 0.0, 1.0)
        return image_scaled

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

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate gram matrices
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or len(
                input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, shape=[-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(tf.transpose(a), a) / tf.cast(n, tf.float32)
        gram = tf.reshape(gram, shape=[1, -1, channels])
        return gram
