import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Set the directories
base_dir = os.path.dirname(os.path.abspath(__file__))

# Wound bed directories
wound_types = ['Abscess', 'Deep tissue', 'Sacral', 'Pressure injury', 'Venous ulcer', 'Uncategorized']
num_wound_types = len(wound_types)
predictions_dir = os.path.join(base_dir, 'Predictions')
train_dir = os.path.join(base_dir, 'Normal')
valid_dir = os.path.join(base_dir, 'Validation')
test_dir = os.path.join(base_dir, 'Test')
mask_dir = os.path.join(base_dir, 'TestM')  # Path to the test mask directory

# Make sure predictions directory exists
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)


# Function to load and preprocess the image
def load_image(img_path, mask=False):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=1 if mask else 3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0  # normalize to [0,1] range
    wound_type = img_path.split(os.sep)[-2]  # get the wound type from the directory name
    return img, wound_type


# Function to augment the image
def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # Rotate image using TensorFlow operations
    image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return image


# Function to get the label (mask) corresponding to the image
def load_labels(image_paths, mask_dir):
    masks = []
    for path in image_paths:
        mask_path = os.path.join(mask_dir, os.path.basename(path))
        if os.path.isfile(mask_path):
            mask = load_image(mask_path, mask=True)[0]  # Load the mask image only
            mask = tf.cast(mask > 0, dtype=tf.float32)  # Convert grayscale mask to binary mask
            masks.append(mask)
        else:
            print(f"Mask file {mask_path} not found.")
    return np.array(masks)


# Get paths for all the images
train_image_paths = []
train_types = []
for wound_type in wound_types:
    wound_type_dir = os.path.join(base_dir, wound_type)
    image_names = os.listdir(wound_type_dir)
    for img_name in image_names:
        img_path = os.path.join(train_dir, img_name)
        train_image_paths.append(img_path)
        train_types.append(wound_type)

valid_image_paths = [os.path.join(valid_dir, img_name) for img_name in os.listdir(valid_dir)]
test_image_paths = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir)]
test_mask_paths = [os.path.join(mask_dir, img_name) for img_name in os.listdir(test_dir)]

# Load all the images and masks
train_images, train_types = zip(*[load_image(path) for path in train_image_paths])
train_labels = load_labels(train_image_paths, os.path.join(base_dir, 'NormalM'))
train_types = [wound_types.index(wound_type) if wound_type in wound_types else wound_types.index('Uncategorized') for wound_type in train_types]  # convert wound types to integers
valid_images = []
for path in valid_image_paths:
    img, _ = load_image(path)
    valid_images.append(img)
valid_images = np.array(valid_images)

valid_labels = load_labels(valid_image_paths, os.path.join(base_dir, 'ValidationM'))

# Load test masks
test_masks = load_labels(test_mask_paths, mask_dir)

# Apply data augmentation to training images
augmented_images = [augment_image(img) for img in train_images]

# Build the MobileNetV2 model
base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=[224, 224, 3])
base_model.trainable = False

inputs = base_model.input

# Classification branch
x1 = base_model.output
x1 = keras.layers.GlobalAveragePooling2D()(x1)
x1 = keras.layers.Dense(num_wound_types, activation='softmax')(x1)

# Segmentation branch
x2 = base_model.output
x2 = keras.layers.UpSampling2D((2, 2))(x2)
x2 = keras.layers.DepthwiseConv2D((3, 3), activation='relu', padding='same')(x2)
x2 = keras.layers.BatchNormalization()(x2)
x2 = keras.layers.Conv2D(128, (1, 1), activation='relu')(x2)
x2 = keras.layers.UpSampling2D((2, 2))(x2)
x2 = keras.layers.DepthwiseConv2D((3, 3), activation='relu', padding='same')(x2)
x2 = keras.layers.BatchNormalization()(x2)
x2 = keras.layers.Conv2D(64, (1, 1), activation='relu')(x2)
x2 = keras.layers.UpSampling2D((2, 2))(x2)
x2 = keras.layers.DepthwiseConv2D((3, 3), activation='relu', padding='same')(x2)
x2 = keras.layers.BatchNormalization()(x2)
x2 = keras.layers.Conv2D(32, (1, 1), activation='relu')(x2)
x2 = keras.layers.UpSampling2D((2, 2))(x2)
x2 = keras.layers.DepthwiseConv2D((3, 3), activation='relu', padding='same')(x2)
x2 = keras.layers.BatchNormalization()(x2)
x2 = keras.layers.Conv2D(16, (1, 1), activation='relu')(x2)
x2 = keras.layers.UpSampling2D((2, 2))(x2)
x2 = keras.layers.DepthwiseConv2D((3, 3), activation='relu', padding='same')(x2)
x2 = keras.layers.BatchNormalization()(x2)
x2 = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x2)

# Define custom dice loss for segmentation
def dice_loss(y_true, y_pred):
    smooth = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice_coefficient = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice_coefficient

model = Model(inputs=inputs, outputs=[x1, x2])  # x1 is the output of the classification branch, x2 is the output of the segmentation branch

model.compile(optimizer=Adam(), loss=['sparse_categorical_crossentropy', dice_loss], loss_weights=[1, 1], metrics=['accuracy', dice_loss])  # you can adjust loss_weights as needed

# Define evaluation metrics
metrics = {
    'classification': ['accuracy'],
    'segmentation': ['accuracy'],
}

# Define learning rate schedule
def lr_schedule(epoch):
    initial_learning_rate = 0.001
    if epoch < 50:
        return initial_learning_rate
    elif epoch < 80:
        return initial_learning_rate * 0.1
    else:
        return initial_learning_rate * 0.01

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Perform k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True)
fold = 1
history_list = []  # Initialize empty list for storing history objects
for train_indices, val_indices in kfold.split(train_images):
    print(f"Training Fold {fold}...")
    fold_train_images = np.array(train_images)[train_indices]
    fold_train_types = np.array(train_types)[train_indices]
    fold_val_images = np.array(train_images)[val_indices]
    fold_val_types = np.array(train_types)[val_indices]
    fold_train_labels = np.array(train_labels)[train_indices]
    fold_val_labels = np.array(train_labels)[val_indices]

    # Apply data augmentation to training images and masks
    fold_augmented_images = [augment_image(img) for img in fold_train_images]
    fold_augmented_labels = [augment_image(mask) for mask in fold_train_labels]

    # Train the model
    history = model.fit(
        np.array(fold_augmented_images),
        [np.array(fold_train_types), np.array(fold_augmented_labels)],
        validation_data=(np.array(fold_val_images), [np.array(fold_val_types), np.array(fold_val_labels)]),
        epochs=100,
        batch_size=16,
        callbacks=[early_stopping, LearningRateScheduler(lr_schedule)],
    )

    # Append the history object to the list
    history_list.append(history)

    # Save the model
    model.save(f"model_fold{fold}.h5")

    # Load the best model for evaluation
    model = keras.models.load_model(f"model_fold{fold}.h5", custom_objects={'dice_loss': dice_loss})
    model.compile(optimizer=Adam(), loss=['sparse_categorical_crossentropy', dice_loss], loss_weights=[1, 1], metrics=['accuracy', dice_loss])

    # Load the test labels
    test_labels_classification = np.array([0 for _ in range(len(test_image_paths))])

    # Evaluate on test images
    test_images, _ = zip(*[load_image(path) for path in test_image_paths])
    test_types_pred, test_labels_pred = model.predict(np.array(test_images))  # Predict masks for the current batch

    accuracy_classification = test_types_pred[0]
    accuracy_segmentation = test_labels_pred[0]

    print(f"Classification Accuracy: {accuracy_classification}")
    print(f"Segmentation Accuracy: {accuracy_segmentation}")

    fold += 1


# Visualize the predicted masks for up to 100 images, 5 at a time
for batch in range(20):  # there are 20 batches in 100 images if each batch has 5 images
    start_index = batch * 5  # starting index for this batch
    end_index = start_index + 5  # ending index for this batch

    # If start_index is greater than or equal to the number of images, break the loop
    if start_index >= len(test_image_paths):
        break

    test_batch_paths = test_image_paths[start_index:end_index]  # Get paths for the current batch
    test_images, _ = zip(*[load_image(path) for path in test_batch_paths])  # Load test images for the current batch
    preds = model.predict(np.array(test_images))  # Predict masks for the current batch

    fig, axes = plt.subplots(nrows=end_index - start_index, ncols=2, figsize=(10, 4 * (end_index - start_index)))  # Adjust figsize as needed

    for i in range(len(test_images)):  # Iterate through the test images in the current batch
        axes[i, 0].imshow(test_images[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        pred_mask = (preds[1][i, :, :, 0] > 0.5).astype(np.uint8)  # Apply threshold to get binary mask
        pred_mask = np.squeeze(pred_mask)  # Remove the single-dimensional entries
        print(f"Shape of predicted mask: {pred_mask.shape}")

        # Displaying the predicted mask
        axes[i, 1].imshow(pred_mask, cmap='gray')
        axes[i, 1].set_title('Predicted Mask')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


# Plot accuracy and loss
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
for history in history_list:
    plt.plot(history.history['classification_accuracy'])
    plt.plot(history.history['val_classification_accuracy'])
plt.title('Classification Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')

plt.subplot(1, 2, 2)
for history in history_list:
    plt.plot(history.history['segmentation_accuracy'])
    plt.plot(history.history['val_segmentation_accuracy'])
plt.title('Segmentation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')

plt.tight_layout()
plt.show()
