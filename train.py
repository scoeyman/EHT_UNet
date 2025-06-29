import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from scipy.ndimage import binary_fill_holes
from math import ceil

from UNet import unet
from callbacks import PredictionPlotter
from tensorflow.keras.utils import img_to_array

# Paths
TRACED_DIR = "data/traced_resized"
UNTRACED_DIR = "data/untraced_resized"
MASK_OUTPUT_DIR = "data/binary_masks"
MODEL_PATH = "models/model.keras"

# Load and preprocess training images
def load_images(image_dir, size=(256, 256)):
    images = []
    for file in os.listdir(image_dir):
        if file.lower().endswith(('.jpg', '.png')):
            img = Image.open(os.path.join(image_dir, file)).resize(size).convert('RGB')
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
    return np.array(images)

# Create binary masks from traced images
def create_masks(traced_dir, output_dir, size=(256, 256)):
    masks = []
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(traced_dir):
        if file.lower().endswith(('.jpg', '.png')):
            img = Image.open(os.path.join(traced_dir, file)).resize(size).convert('RGB')
            arr = img_to_array(img)
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            mask = np.where((g >= 175) & (g > r) & (g > b) & (r <= 125) & (b <= 125), 1, 0)
            filled = binary_fill_holes(mask).astype(np.uint8) * 255
            output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_mask.png")
            cv2.imwrite(output_path, filled)
            masks.append(filled / 255.0)
    return np.expand_dims(np.array(masks), axis=-1)

# Data Generators
def get_data_generators(images, masks):
    image_gen = ImageDataGenerator(rotation_range=45, width_shift_range=0.075, height_shift_range=0.075,
                                   shear_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
    label_gen = ImageDataGenerator(rotation_range=45, width_shift_range=0.075, height_shift_range=0.075,
                                   shear_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
    image_gen.fit(images, seed=1)
    label_gen.fit(masks, seed=1)
    return image_gen.flow(images, batch_size=20, seed=1), label_gen.flow(masks, batch_size=20, seed=1)

# Main Training Loop
def main():
    print("Loading data...")
    images = load_images(UNTRACED_DIR)
    masks = create_masks(TRACED_DIR, MASK_OUTPUT_DIR)

    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=32)

    image_gen, label_gen = get_data_generators(X_train, y_train)
    train_gen = zip(image_gen, label_gen)

    model = unet()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        PredictionPlotter(model, X_val, y_val, image_gen, label_gen, seed=1)
    ]

    print("Training model...")
    model.fit(train_gen, epochs=200, validation_data=(X_val, y_val), steps_per_epoch=ceil(len(X_train)/6), callbacks=callbacks)

if __name__ == "__main__":
    main()
