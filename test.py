import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Set the directories
base_dir = 'C:\\Users\\dperk\\OneDrive\\Desktop'
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
    masks = [load_image(mask_path, mask=True) for mask_path in mask_paths]
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

# Build the U-Net model with a pre-trained VGG16 as the encoder
base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=[224, 224, 3])
base_model.trainable = False

inputs = base_model.input
x = base_model.output
x = keras.layers.UpSampling2D((2,2))(x)  # upsample the feature map
x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)  # convolve to reduce the number of filters
x = keras.layers.UpSampling2D((2,2))(x)
x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2,2))(x)
x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2,2))(x)
x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2,2))(x)
x = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)



model = Model(inputs=inputs, outputs=x)

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, validation_data=(valid_images, valid_labels), epochs=20, batch_size=16)

# Load test images
test_images = np.array([load_image(path) for path in test_image_paths])

# Generate predictions on test images
preds = model.predict(test_images)

# Save predicted masks to the 'predictions' directory
for i in range(len(preds)):
    pred_mask = preds[i]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # apply a threshold to get binary mask
    pred_mask = tf.image.encode_jpeg(pred_mask)  # convert to JPEG format
    img_name = os.path.basename(test_image_paths[i])
    pred_mask_path = os.path.join(predictions_dir, img_name)
    tf.io.write_file(pred_mask_path, pred_mask)

# plot accuracy and loss
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')

plt.tight_layout()
plt.show()