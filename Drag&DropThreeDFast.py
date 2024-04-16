import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import messagebox

# Your previously provided processing functions and logic
def load_image(img_path):
    img = Image.open(img_path)
    original_size = img.size
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return img, original_size

def process_files(files):
    jpg_files = [f for f in files if f.endswith('.jpg')]
    obj_files = [f for f in files if f.endswith('.obj')]
    mtl_files = [f for f in files if f.endswith('.mtl')]

    if not (jpg_files and obj_files and mtl_files):
        messagebox.showerror("Invalid Files", "Please drag and drop valid 3D files (.jpg, .mtl, .obj).")
        return

    model = tf.keras.models.load_model("model.h5")

    # Keeping track of original sizes
    images = []
    original_sizes = []

    for path in jpg_files:
        img, original_size = load_image(path)
        images.append(img)
        original_sizes.append(original_size)

    test_images = np.array(images)
    preds = model.predict(test_images)

    output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Processed")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (img, pred, original_size) in enumerate(zip(test_images, preds, original_sizes)):
        masked_pred = np.where(pred.squeeze() < 0.3, np.nan, pred.squeeze())

        plt.imshow(img)
        plt.imshow(masked_pred, alpha=0.5, cmap='jet')
        plt.axis('off')
        output_path = os.path.join(output_dir, f"processed_{idx}.jpg")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Resize to original dimensions
        original_img = Image.open(output_path)
        original_img = original_img.resize(original_size)
        original_img.save(output_path)

    messagebox.showinfo("Success", f"Files processed and saved in {output_dir}")

def on_drop(event):
    files = event.data.strip().split()
    process_files(files)

root = TkinterDnD.Tk()
root.title('3D File Processor')

label = tk.Label(root, text="Drag & Drop 3D files (.jpg, .mtl, .obj) here", padx=10, pady=10, bg="lightgray")
label.pack(pady=(50, 0), padx=50, expand=True)

root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_drop)

root.geometry("400x200")
root.mainloop()
