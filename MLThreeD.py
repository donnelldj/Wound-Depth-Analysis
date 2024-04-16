import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Set the directories
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, 'Normal')
valid_dir = os.path.join(base_dir, 'Validation')
test_dir = os.path.join(base_dir, 'Test')
predictions_dir = os.path.join(base_dir, 'Predictions')

# Make sure predictions directory exists
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)

# Function to load and preprocess the image
def load_image(img_path, mask=False):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=1 if mask else 3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0  # normalize to [0,1] range
    return img

# Function to get the label (mask) corresponding to the image
def load_labels(image_paths, mask_dir):
    mask_paths = [os.path.join(mask_dir, os.path.basename(path)) for path in image_paths]
    masks = []
    for mask_path in mask_paths:
        if os.path.isfile(mask_path):
            masks.append(load_image(mask_path, mask=True))
        else:
            print(f"Mask file {mask_path} not found.")
    return np.array(masks)

# Get paths for all the images
train_image_paths = [os.path.join(train_dir, img_name) for img_name in os.listdir(train_dir)]
valid_image_paths = [os.path.join(valid_dir, img_name) for img_name in os.listdir(valid_dir)]
test_image_paths = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir)]

# Load all the images and masks
train_images = np.array([load_image(path) for path in train_image_paths])
train_labels = load_labels(train_image_paths, os.path.join(base_dir, 'NormalM'))
valid_images = np.array([load_image(path) for path in valid_image_paths])
valid_labels = load_labels(valid_image_paths, os.path.join(base_dir, 'ValidationM'))

# U-Net model
def build_unet_model(input_shape=(224, 224, 3)):
    inputs = keras.Input(input_shape)
    
    # Contracting Path
    c1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    
    c2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    
    c4 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)
    
    c5 = keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Expansive Path
    u6 = keras.layers.UpSampling2D((2, 2))(c5)
    u6 = keras.layers.Concatenate()([u6, c4])
    c6 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = keras.layers.UpSampling2D((2, 2))(c6)
    u7 = keras.layers.Concatenate()([u7, c3])
    c7 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = keras.layers.UpSampling2D((2, 2))(c7)
    u8 = keras.layers.Concatenate()([u8, c2])
    c8 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = keras.layers.UpSampling2D((2, 2))(c8)
    u9 = keras.layers.Concatenate()([u9, c1])
    c9 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    return Model(inputs=[inputs], outputs=[outputs])

# Create the U-Net model
model = build_unet_model()

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, validation_data=(valid_images, valid_labels), epochs=20, batch_size=16)

# Save the model
model.save("model.h5")

# Load the model
model = keras
