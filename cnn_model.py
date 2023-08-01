# cnn_model.py

from tensorflow.keras import models, optimizers
import cnn_layers

def build_cnn(input_shape=(28, 28, 1), pooling_type='max', optimizer_type='adam', feature_classification='dense'):
    model = models.Sequential()
    
    # Convolutional Layer
    model.add(cnn_layers.convolutional_layer(input_shape=input_shape))
    
    # Pooling Layer
    model.add(cnn_layers.pooling_layer(pooling_type))
    
    # Additional Convolutional Layer
    model.add(cnn_layers.convolutional_layer())
    
    # Flatten Layer
    model.add(cnn_layers.flatten_layer())
    
    # Feature Classification Layer
    if feature_classification == 'dense':
        model.add(cnn_layers.dense_layer())
    else:
        raise ValueError("Invalid feature_classification type. Currently only 'dense' is supported.")
    
    # Output Layer
    model.add(cnn_layers.output_layer())
    
    # Compile the model with the specified optimizer
    if optimizer_type == 'adam':
        optimizer = optimizers.Adam()
    elif optimizer_type == 'sgd':
        optimizer = optimizers.SGD()
    else:
        raise ValueError("Invalid optimizer_type. Choose 'adam' or 'sgd'.")
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
