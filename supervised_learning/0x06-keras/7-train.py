#!/usr/bin/env python3
"""
Trains a model using mini-batch gradient descent
based on 6-train.py
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True):
    """
    Returns: Nothing
    """
    def schedule(epoch):
        return 1 / (1 + decay_rate * epochs) + alpha
        
        
    if validation_data is not None:
        callbacks = [K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience,
                                               verbose=verbose),
                     K.callbacks.LearningRateScheduler(schedule)]
        
    
    network.fit(data,
                labels,
                batch_size,
                epochs,
                validation_data=validation_data,
                callbacks=callbacks)
