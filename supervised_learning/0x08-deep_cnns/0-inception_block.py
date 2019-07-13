#!/usr/bin/env python3
"""
Builds an inception block
"""
import tensorflow.keras as K



def inception_block(A_prev, filters):
    """
    Returns: the concatenated output of the inception block
    """

    tower_0 = K.layers.Conv2D(filters[0], (1,1), padding="same", activation='relu')(A_prev)
    tower_1 = K.layers.Conv2D(filters[1], (1,1), padding="same", activation='relu')(A_prev)
    tower_1 = K.layers.Conv2D(filters[2], (3,3), padding='same', activation='relu')(tower_1)


    tower_2 = K.layers.Conv2D(filters[3], (1,1), padding='same', activation='relu')(A_prev)
    tower_2 =  K.layers.Conv2D(filters[4], (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = K.layers.MaxPooling2D((3,3), (1,1), padding='same')(A_prev)
    tower_3 = K.layers.Conv2D(filters[5], (1,1), padding='same', activation='relu')(tower_3)

    output = K.layers.concatenate([tower_0, tower_1, tower_2, tower_3])

    return output
