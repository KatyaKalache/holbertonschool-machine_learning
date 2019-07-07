#!/usr/bin/python3
"""
Builds a modified version of the LeNet-5 architecture 
using tensorflow
"""
import tensorflow as tf


def lenet5(x, y):
    """
    Returns:
    a tensor for the softmax activated output
    a training operation that utilized Adam optimization (with default hyperparameters)
    a tensor for the loss of the netowrk
    a tensor for the accuracy of the network
    """

    conv_layer_1 = tf.layers.Conv2D(filters=6,
                                    kernel_size=(5,5),
                                    padding='same',
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'))(x)

    # Activation
    conv_layer_1 = tf.nn.relu(conv_layer_1)

    #Pooling
    conv_layer_1 = tf.layers.MaxPooling2D(pool_size=(2,2),
                                          strides=(2,2))(conv_layer_1)

    # initializing layer 2
    conv_layer_2 = tf.layers.Conv2D(filters=16,
                                    kernel_size=(5,5),
                                    padding='valid',
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'))(conv_layer_1)

    # Activation
    conv_layer_2 = tf.nn.relu(conv_layer_2)

    # Pooling
    conv_layer_2 = tf.layers.MaxPooling2D(pool_size=(2,2),
                                          strides=(2,2))(conv_layer_2)

    # Flatten
    flatten = tf.layers.Flatten()(conv_layer_2)

    # Fully connected
    fully_con_1 = tf.contrib.layers.fully_connected(inputs = flatten,
                                                    num_outputs=120,
                                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'))

    # Activation
    fully_con_1 = tf.nn.relu(fully_con_1)

    fully_con_2 = tf.contrib.layers.fully_connected(inputs=fully_con_1,
                                                    num_outputs=80,
                                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'))
    fully_con_2 = tf.nn.relu(fully_con_2)

    fully_con_soft_max = tf.contrib.layers.fully_connected(inputs=fully_con_2,
                                                    num_outputs=10,
                                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'))
    # Activtion with softMax
    fully_con_soft_max = tf.nn.softmax(fully_con_soft_max)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y,
                                           logits=fully_con_soft_max)
    
    train_op = tf.train.AdamOptimizer().minimize(loss)


    pred = tf.equal(tf.argmax(fully_con_soft_max, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
    
    return fully_con_soft_max, train_op, loss, accuracy
