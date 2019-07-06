#!/usr/bin/env python3
"""
Trains a model using mini-batch gradient descent
based on 7-train.py
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True):
    """
    Returns: Nothing
    """
    
    
    def schedule(epoch):
        return 1 / (1 + decay_rate * epochs) + alpha
    callbacks = []
    if validation_data is not None:
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience,
                                                   verbose=verbose))
        if learning_rate_decay:
            callbacks.append(K.callbacks.LearningRateScheduler(schedule))

        if save_best:
            checkpointer = K.callbacks.ModelCheckpoint(filepath=filepath, verbose=verbose, save_best_only=True)
            callbacks.append(checkpointer)
    network.fit(data,
                labels,
                batch_size,
                epochs,
                validation_data=validation_data,
                callbacks=callbacks)
