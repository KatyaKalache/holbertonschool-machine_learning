#!/usr/bin/env python3
"""
Trains a model using mini-batch gradient descent
based on 5-train.py
"""
import tensorflow.keras as K

def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True):
    """
    Returns: Nothing
    """
    if validation_data is not None:
        es = [K.callbacks.EarlyStopping(monitor='val_loss',
                                        patience=patience,
                                        verbose=verbose)]
    es = []
    network.fit(data,
                labels,
                batch_size,
                epochs,
                validation_data=validation_data,
                callbacks=es)

    
