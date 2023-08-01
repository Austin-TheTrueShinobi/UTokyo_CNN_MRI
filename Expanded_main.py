import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, 
                                     TimeDistributed, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Load MRI data (dummy data for this example)
mri_data = np.random.rand(100, 128, 128, 3)  # 100 MRI images of size 128x128 with 3 channels
labels = np.random.randint(2, size=100)  # Binary labels for the MRI data

# Split data into training and testing sets
train_data = mri_data[:80]
test_data = mri_data[80:]
train_labels = labels[:80]
test_labels = labels[80:]

# Build the LSTM-CNN model
def build_lstm_cnn_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # CNN layers
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    x = TimeDistributed(Flatten())(x)
    
    # LSTM layer
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = LSTM(64)(x)
    
    # Dense layer for classification
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_lstm_cnn_model((10, 128, 128, 3))
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(train_data, train_labels, epochs=20, validation_data=(test_data, test_labels), 
                    callbacks=[early_stopping, checkpoint])

# Visualization functions
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    
    plt.show()

def visualize_filters(layer_name):
    layer = model.get_layer(name=layer_name)
    filters, biases = layer.get_weights()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    n_filters, ix = 6, 1
    for i in range(n_filters):
        f = filters[:, :, :, i]
        for j in range(3):
            ax = plt.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(f[:, :, j], cmap='viridis')
            ix += 1
    plt.show()

def visualize_feature_maps(layer_name, input_data):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(input_data)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(intermediate_output[0, :, :, i], cmap='viridis')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_training_history(history)
visualize_filters('conv2d_1')
visualize_feature_maps('max_pooling2d_1', np.expand_dims(train_data[0], axis=0))
