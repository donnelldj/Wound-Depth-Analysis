import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set the paths to the images folder and the masks folder
images_path = "C:\\Users\\dperk\\OneDrive\\Desktop\\Normal"
masked_path = "C:\\Users\\dperk\\OneDrive\\Desktop\\Masked"

# Create the masks folder if it doesn't exist
if not os.path.exists(masked_path):
    os.makedirs(masked_path)

# Load the image filenames
image_filenames = os.listdir(images_path)

# Loop through the images and mask them
for i, filename in enumerate(image_filenames):
    # Load the image
    image_path = os.path.join(images_path, filename)
    image = cv2.imread(image_path)

    # Create a copy of the image to draw on
    drawing = image.copy()

    # Create a window to display the image
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", (800, 800))
    cv2.imshow("Image", drawing)

    # Create a copy of the original image for highlighting the drawing
    highlight = image.copy()

    # Set up the mouse callback function for drawing
    pts = []
    drawing_done = False

    def draw(event, x, y, flags, param):
        global pts, drawing_done
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            pts.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing_done = True

    cv2.setMouseCallback("Image", draw)

    # Wait for the user to finish drawing
    while not drawing_done:
        # Draw the current polygon on the highlight image
        if len(pts) > 1:
            cv2.polylines(highlight, [np.array(pts)],
                          True, (0, 255, 255), thickness=5)

        # Merge the original and highlighted images
        output = cv2.addWeighted(image, 0.7, highlight, 0.3, 0)

        cv2.imshow("Image", output)
        cv2.waitKey(1)

    # Create a mask image for the wound region
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Draw the final polygon on the mask
    if len(pts) > 2:
        cv2.fillPoly(mask, [np.array(pts)], 255)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Save the masked image to the masks folder
    mask_path = os.path.join(masked_path, filename)
    cv2.imwrite(mask_path, masked_image)

    # Rename the original image
    os.rename(image_path, os.path.join(images_path, f"image_{i+1}.jpg"))

    # Close the plot window
    plt.close()

print("Masking complete.")
