import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil
from pywavefront import visualization, Wavefront

# Set the directory for test images and 3D files
base_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(base_dir, '3D')
output_dir = os.path.join(test_dir, 'Processed')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to load and preprocess the image
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0  # normalize to [0,1] range
    return img

# Separate the different file types
jpg_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
obj_files = [f for f in os.listdir(test_dir) if f.endswith('.obj')]
mtl_files = [f for f in os.listdir(test_dir) if f.endswith('.mtl')]

# Copy .obj and .mtl files to the output directory
for f in obj_files + mtl_files:
    shutil.copy2(os.path.join(test_dir, f), output_dir)

# Process .jpg images
test_image_paths = [os.path.join(test_dir, img_name) for img_name in jpg_files]

# Load the model
model = tf.keras.models.load_model("model.h5")

# Load test images
test_images = np.array([load_image(path) for path in test_image_paths])

# Generate predictions on test images
preds = model.predict(test_images)

# Mask the predictions below the threshold
masked_preds = np.where(preds < 0.3, np.nan, preds)

# Visualize and save the processed images with the overlay
for i, img_path in enumerate(test_image_paths):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(test_images[i])
    ax.imshow(masked_preds[i].squeeze(), alpha=0.5, cmap='jet') 
    ax.axis('off')

    output_img_path = os.path.join(output_dir, os.path.basename(img_path))
    plt.tight_layout()
    plt.savefig(output_img_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

print(f"Processed images saved in {output_dir}")

# Load and display the 3D model in the viewer
for obj in obj_files:
    obj_path = os.path.join(test_dir, obj)
    scene = Wavefront(obj_path, create_materials=True, collect_faces=True)
    visualization.draw(scene)
