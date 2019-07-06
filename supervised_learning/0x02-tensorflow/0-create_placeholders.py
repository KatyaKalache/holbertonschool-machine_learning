#!/usr/bin/env python3
"""
Function that that returns two placeholders
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Returns: placeholders named x and y
    """
    x = tf.placeholder(name='x', shape=(None, nx), dtype=tf.float32)
    y = tf.placeholder(name='y', shape=(None, classes), dtype=tf.float32)

    return x, y
