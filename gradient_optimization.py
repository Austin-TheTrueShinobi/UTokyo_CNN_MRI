import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def build_cnn_model(input_shape):
    """
    Build a CNN model for MRI data classification.
    
    Parameters:
    - input_shape: Shape of the input MRI data.
    
    Returns:
    - model: Compiled CNN model.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # Assuming binary classification
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, mri_data, labels, epochs=10):
    """
    Train the CNN model on MRI data.
    
    Parameters:
    - model: Compiled CNN model.
    - mri_data: MRI data as a numpy array.
    - labels: Ground truth labels for the MRI data.
    - epochs: Number of training epochs.
    
    Returns:
    - history: Training history.
    """
    history = model.fit(mri_data, labels, epochs=epochs)
    return history

# Example usage:
input_shape = (mri_data.shape[1], mri_data.shape[2], 1)
model = build_cnn_model(input_shape)
history = train_model(model, mri_data[..., np.newaxis], labels)
