import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set the directory for test images
base_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(base_dir, 'Test')

# Function to load and preprocess the image
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0  # normalize to [0,1] range
    return img

# Get paths for all the test images
test_image_paths = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir)]

# Load the model
model = tf.keras.models.load_model("model1.h5")

# Load test images
test_images = np.array([load_image(path) for path in test_image_paths])

# Generate predictions on test images
preds = model.predict(test_images)

# Visualize the predictions for up to 100 images, 5 at a time
for batch in range(20):
    start_index = batch * 5
    end_index = start_index + 5
    
    # Adjust if the total number of images is less than 100
    if start_index >= len(test_images):
        break
    if end_index > len(test_images):
        end_index = len(test_images)

    fig, axes = plt.subplots(nrows=end_index-start_index, ncols=2, figsize=(10, 4*(end_index-start_index)))

    for i in range(start_index, end_index):
        axes[i - start_index, 0].imshow(test_images[i])
        axes[i - start_index, 0].set_title('Original Image')
        axes[i - start_index, 0].axis('off')
        axes[i - start_index, 1].imshow(test_images[i], cmap='gray')
        axes[i - start_index, 1].imshow(preds[i].squeeze(), alpha=0.5, cmap='jet') # using squeeze() to remove the channel dimension
        axes[i - start_index, 1].set_title('Predicted Mask')
        axes[i - start_index, 1].axis('off')

    plt.tight_layout()
    plt.show()
