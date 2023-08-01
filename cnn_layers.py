# cnn_layers.py

import tensorflow as tf
from tensorflow.keras import layers

def convolutional_layer(filters=32, kernel_size=(3, 3), activation='relu'):
    return layers.Conv2D(filters, kernel_size, activation=activation)

def pooling_layer(pooling_type='max', pool_size=(2, 2)):
    if pooling_type == 'max':
        return layers.MaxPooling2D(pool_size)
    elif pooling_type == 'average':
        return layers.AveragePooling2D(pool_size)
    else:
        raise ValueError("Invalid pooling_type. Choose 'max' or 'average'.")

def flatten_layer():
    return layers.Flatten()

def dense_layer(units=64, activation='relu'):
    return layers.Dense(units, activation=activation)

def output_layer(units=10, activation='softmax'):
    return layers.Dense(units, activation=activation)
