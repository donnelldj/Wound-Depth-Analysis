import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
# This is a placeholder - replace with your own data loading mechanism
# Images should be normalized (pixel values from 0 to 1) and labels should be one-hot encoded
(train_images, train_labels), (test_images, test_labels) = (None, None), (None, None)

# Create a Sequential model
model = Sequential()

# Add a Conv2D layer with 32 filters, a 3x3 kernel, and 'relu' activation function
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3))) # assuming you're working with 64x64 RGB images
model.add(MaxPooling2D((2, 2)))

# Add another Conv2D layer with 64 filters and a 3x3 kernel
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the tensor output from the previous layer
model.add(Flatten())

# Add a Dense layer with 64 units and 'relu' activation function
model.add(Dense(64, activation='relu'))

# Add output layer with 2 units (for 'wound' and 'no wound') and 'softmax' activation function
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# Save the model for future use
model.save('wound_detection_model.h5')
