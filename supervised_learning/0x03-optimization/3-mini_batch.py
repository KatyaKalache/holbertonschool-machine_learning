#!/usr/bin/env python3
"""
Trains a loaded neural network model using mini-batch gradient descent
"""
import numpy as np
import tensorflow as tf


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """
    Returns: the path where the model was saved
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x_tensor = tf.get_collection('x')[0]
        y_tensor = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        mini_batches = X_train.shape[0] // batch_size
        
        for i in range(epochs+1):
            # shuffle input data
            np.random.permutation(X_train)
            np.random.permutation(Y_train)
            print ("After {} epochs:".format(i))
            train_cost = sess.run([loss], feed_dict = {x_tensor: X_train, y_tensor: Y_train})
            print("\tTraining Cost: {}".format(train_cost))
            train_accuracy = sess.run([accuracy], feed_dict = {x_tensor: X_train, y_tensor: Y_train})
            print ("\tTraining Accuracy: {}".format(train_accuracy))
            valid_cost = sess.run([loss], feed_dict = {x_tensor: X_valid, y_tensor: Y_valid})
            print("\tValidation Cost: {}".format(valid_cost))
            valid_accuracy = sess.run([accuracy], feed_dict = {x_tensor: X_valid, y_tensor: Y_valid})
            print("\tValidation Accuracy: {}".format(valid_accuracy))
         
            for each_batch in range(mini_batches+1):
                batch_start = each_batch * batch_size
                batch_end = (each_batch+1) * batch_size
                sess.run(train_op, feed_dict={x_tensor: X_train[batch_start:batch_end],
                                              y_tensor: Y_train[batch_start:batch_end]})
                if each_batch % 100 == 0 and each_batch != 0 and i != epochs:
                    print('\tStep {}'.format(each_batch))
                    step_cost = sess.run([loss], feed_dict = {x_tensor: X_train[batch_start:batch_end],
                                                              y_tensor: Y_train[batch_start:batch_end]})
                    print('\t\tCost: {}'.format(step_cost))
                    step_accuracy = sess.run([accuracy], feed_dict = {x_tensor: X_train[batch_start:batch_end],
                                                                      y_tensor: Y_train[batch_start:batch_end]})
                    print('\t\tAccuracy: {}'.format(step_accuracy))

        return saver.save(sess, save_path)
