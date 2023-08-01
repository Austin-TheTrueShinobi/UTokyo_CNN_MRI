import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

species = ("MIN Pooling", "MAX Pooling")

means = {
    'Batch Size and TTL': (18, 19, 20),
    'Test Loss': (20, 19, 22),
    'Test Accuracty': (23, 13, 19),
        }

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# build the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

# train the model
batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)



# evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])



x = np.arange(len(species))  
# the label locations 
width = 0.25  # the width of the bars 
multiplier = 0  
fig, ax = plt.subplots(layout='constrained')  
for attribute, measurement in means.items():     
    offset = width * multiplier     
    rects = ax.bar(x + offset, measurement, width, label=attribute)     
    ax.bar_label(rects, padding=3)     
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc. 
ax.set_ylabel('Length (mm)') 
ax.set_title('Penguin attributes by species') 
ax.set_xticks(x + width, species) 
ax.legend(loc='upper left', ncols=3) 
ax.set_ylim(0, 250)  

plt.show()







