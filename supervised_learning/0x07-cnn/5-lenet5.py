#!/usr/bin/python3
"""
Builds a modified version of the LeNet-5 architecture using keras
"""
import tensorflow.keras as K


def lenet5(X):
    """
    Returns: a K.Model compiled to use Adam optimization 
    (with default hyperparameters) and accuracy metrics
    """
    conv_layer_1 = K.layers.Conv2D(filters=6,
                    kernel_size=(5,5),
                    padding="same",
                    kernel_initializer="he_normal",
                    activation="relu")(X)
    
    pooled_layer_1 = K.layers.MaxPool2D(pool_size=(2,2),
                                        strides=(2,2))(conv_layer_1)
    conv_layer_2 = K.layers.Conv2D(filters=16,
                                   kernel_size=(5,5),
                                   padding="valid",
                                   kernel_initializer="he_normal",
                                   activation="relu")(pooled_layer_1)

    
    pooled_layer_2 =  K.layers.MaxPool2D(pool_size=(2,2),
                                         strides=(2,2))(conv_layer_2)

    pooled_layer_2 = K.layers.Flatten()(pooled_layer_2)
    fully_con_1 = K.layers.Dense(120,
                                 activation="relu",
                                 kernel_initializer="he_normal")(pooled_layer_2)

    fully_con_2 = K.layers.Dense(84,
                                 activation="relu",
                                 kernel_initializer="he_normal")(fully_con_1)

    fully_con_soft = K.layers.Dense(10,
                                    activation="softmax",
                                    kernel_initializer="he_normal")(fully_con_2)

    model = K.Model(inputs=X, outputs=fully_con_soft)
    model.compile(loss="categorical_crossentropy",
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy']) 

    return model
