#!/usr/bin/env python3
"""
Trains a convolutional neural network to classify the CIFAR 100 dataset
"""
import tensorflow.keras as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image
from scipy.misc import imresize

def preprocess_data(X,Y):
    # load dataset
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    # one hot encoding of target values: 10 classes, each class represented by unique integer
    #Y_train = to_categorical(Y_train, 10)
    #Y = to_categorical(Y_test, 10)
    # convert from integers to floats
 #   model = ResNet50(input_shape=(197,197,3), include_top=False, weights='imagenet')
    #Reshaping the training data
    X_train_new = np.array([imresize(X_train[i], (197, 197, 3)) for i in range(0, len(X_train)//3)]).astype('float32')
    np.save("0.npy", X_train_new)
    
    #Preprocessing the data, so that it can be fed to the pre-trained ResNet50 model.
  #  resnet_train_input = preprocess_input(X_train_new)
#    X_test_new = np.array([imresize(X_test[i], (197, 197, 3)) for i in range(0, 4)]).astype('float32')
   # restnet_test_input = preprocess_input(X_test_new)
#    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 #   model.fit(X_train, Y_train, epochs=1)
  #  model.save('cifar10.h5')
   # return X, Y
    
