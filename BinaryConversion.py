import os
import cv2
import numpy as np

# Define the source directory
src_dir = "C:\\Users\\dperk\\OneDrive\\Desktop\\Masked"

# Loop through each file in the source directory
for filename in os.listdir(src_dir):
    # Load the image
    src_path = os.path.join(src_dir, filename)
    image = cv2.imread(src_path)

    # Convert all non-black (not exactly [0, 0, 0]) pixels to white
    image[np.sum(image, axis=-1) > 0] = [255, 255, 255]

    # Overwrite the original image with the modified image
    cv2.imwrite(src_path, image)
