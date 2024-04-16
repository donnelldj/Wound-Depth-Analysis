from tensorflow import keras
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np

# Define custom dice loss for segmentation
def dice_loss(y_true, y_pred):
    smooth = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice_coefficient = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice_coefficient

# Set the directories
base_dir = 'C:\\Users\\dperk\\OneDrive\\Desktop'

# Load the saved model
model = keras.models.load_model("model_fold3.h5", custom_objects={'dice_loss': dice_loss})

# Function to load and preprocess the image
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0  # normalize to [0,1] range
    return img

# Get paths for all the images
test_image_paths = [os.path.join(base_dir, 'Test', img_name) for img_name in os.listdir(os.path.join(base_dir, 'Test'))]

# Load the test images
test_images = [load_image(path) for path in test_image_paths]

# Evaluate on test images
test_labels_pred = [model.predict(np.expand_dims(image, axis=0)) for image in test_images]  # Predict masks for each image

# Visualize the predictions
for i in range(len(test_images)):  # for each image
    image = test_images[i]
    pred_mask = test_labels_pred[i]  # Get the predicted mask for the i-th image
    print(f"Shape of prediction for image {i}: {pred_mask.shape}")  # Print the shape of the prediction

    # Apply thresholding or use different visualization techniques
    threshold = 0.5
    binary_mask = np.where(pred_mask > threshold, 1, 0)

    # Apply the mask as an overlay
    overlay = np.copy(image)
    overlay = np.stack([overlay]*3, axis=-1)  # Convert grayscale to RGB
    overlay[binary_mask == 1] = [0, 255, 0]  # Highlight the mask region in green (adjust colors as desired)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))  # Adjust figsize as needed

    axes[0].imshow(image, cmap='gray')  # show the image in grayscale
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(overlay)
    axes[1].set_title('Predicted Mask Highlighted')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
