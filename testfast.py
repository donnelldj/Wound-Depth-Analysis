import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Set the directories
base_dir = 'C:\\Users\\dperk\\OneDrive\\Desktop'
test_dir = os.path.join(base_dir, 'Test')
predictions_dir = os.path.join(base_dir, 'Predictions')

# Function to load and preprocess the image
def load_image(img_path, mask=False):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=1 if mask else 3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0  # normalize to [0,1] range
    return img

# Get paths for all the test images
test_image_paths = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir)]

# Load test images
test_images = np.array([load_image(path) for path in test_image_paths])

# Load the model
model = keras.models.load_model("model.h5")

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

# Visualize the predictions
for i in range(len(test_images)):  
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_images[i])
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(test_images[i], cmap='gray')
    plt.imshow(preds[i], alpha=0.5, cmap='jet')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
