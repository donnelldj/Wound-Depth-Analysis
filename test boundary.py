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

import skimage

# Visualize the predictions for up to 100 images, 5 at a time
for batch in range(20):  # there are 20 batches in 100 images if each batch has 5 images
    start_index = batch * 5  # starting index for this batch
    end_index = start_index + 5  # ending index for this batch

    # If start_index is greater than or equal to the number of images, break the loop
    if start_index >= len(test_images):
        break

    fig, axes = plt.subplots(nrows=end_index - start_index, ncols=3, figsize=(12, 4 * (end_index - start_index)))  # Adjust figsize as needed

    for i in range(start_index, end_index):  # select a set of 5 images
        original_image = test_images[i]
        predicted_mask = preds[i]
        boundary = skimage.measure.find_contours(predicted_mask, 0.5)

        axes[i - start_index, 0].imshow(original_image)
        axes[i - start_index, 0].set_title('Original Image')
        axes[i - start_index, 0].axis('off')

        axes[i - start_index, 1].imshow(original_image, cmap='gray')
        axes[i - start_index, 1].imshow(predicted_mask, alpha=0.5, cmap='jet')
        axes[i - start_index, 1].set_title('Predicted Mask')
        axes[i - start_index, 1].axis('off')

        axes[i - start_index, 2].imshow(original_image, cmap='gray')
        for contour in boundary:
            axes[i - start_index, 2].plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
        axes[i - start_index, 2].set_title('Predicted Mask with Boundary')
        axes[i - start_index, 2].axis('off')

    plt.tight_layout()
    plt.show()
