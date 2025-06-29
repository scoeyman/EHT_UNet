# --- Import required libraries ---
import os
import keras
import cv2
import csv
import numpy as np
from tensorflow.keras.utils import img_to_array, load_img
from skimage import morphology
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d

# Initialize data storage lists
length_list = []
test_images = []
OG_images = []
distances_list = []

# Set folder paths
main_folder = r""       # Folder containing the test images
current_dir = r""       # Folder containing the trained model and where output will be saved

# Load trained segmentation model
model_path = os.path.join(current_dir, 'model.keras')
model = load_model(model_path)

# Load and preprocess test images
test_images = []
OG_images = []
test_img = []
filenames = []

# Iterate through files in the folder
for file in os.listdir(main_folder):
    if file.startswith('day') and (file.endswith('.jpg') or file.endswith('.png')):
        current_image_name = file
        filenames.append(current_image_name)
        t_frame = Image.open(os.path.join(main_folder, file))
        print("Successfully loaded image:", file)

        # Resize image to model input size and normalize
        img = t_frame.resize((256, 256))
        OG_images.append(img)
        img_array = img_to_array(img) / 255.0
        test_img.append(img_array)

test_images = np.array(test_img)

# Predict segmentation masks using the model 
predictions = model.predict(test_images)

# Process each prediction 
for i in range(len(predictions)):
    current_file = filenames[i]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

    # Convert prediction to binary mask using threshold
    binary_prediction = (predictions[i] > 0.4).astype(np.uint8)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (assumed to be the tissue area)
        largest_contour = max(contours, key=cv2.contourArea)
        contour_frame = np.array(OG_images[i].convert('RGB'), dtype=np.uint8)

        # Draw contour on original image
        cv2.drawContours(contour_frame, [largest_contour], -1, (0, 255, 0), 1)
        ax1.imshow(contour_frame[::-1])  # Display flipped image
        ax1.invert_yaxis()

        # Extract and preprocess contour points
        contour = np.squeeze(largest_contour)
        x_coords, y_coords = contour[:, 0], contour[:, 1]
        max_OG_y = np.max(y_coords)
        y_coords = np.max(y_coords) - y_coords  # Flip Y-axis for processing
        new_contour = np.column_stack((x_coords, y_coords))

        # Smooth the contour points to remove noise
        smoothed_contour = gaussian_filter1d(new_contour, sigma=0.2, axis=0)

        # Separate X and Y for interpolation
        x_values = smoothed_contour[:, 0]
        y_values = smoothed_contour[:, 1]

        # Interpolate contour to get uniform point spacing
        distances = np.sqrt(np.diff(x_values)**2 + np.diff(y_values)**2)
        cumulative_distances = np.cumsum(np.insert(distances, 0, 0))
        interp_cumulative_distances = np.linspace(0, cumulative_distances[-1], 2000)
        interp_x_values = np.interp(interp_cumulative_distances, cumulative_distances, x_values)
        interp_y_values = np.interp(interp_cumulative_distances, cumulative_distances, y_values)

        # Plot smoothed and interpolated contours
        ax2.plot(x_values, y_values, 'o', label='Original Points')
        ax2.plot(interp_x_values, interp_y_values, 'k-', label='Interpolated Points')
        ax3.plot(interp_x_values, interp_y_values, 'k-', label='Interpolated Points')

        # Sort coordinates for measurement extraction
        sorted_indices = np.argsort(interp_y_values)
        sorted_x_values = np.round(interp_x_values[sorted_indices], 1)
        sorted_y_values = np.round(interp_y_values[sorted_indices], 1)

        sorted_indices_x = np.argsort(interp_x_values)
        sorted_x_values_x = np.round(interp_x_values[sorted_indices_x], 1)
        sorted_y_values_x = np.round(interp_y_values[sorted_indices_x], 1)

        # Calculate full length and width of tissue
        A = np.max(sorted_y_values)
        B = np.min(sorted_y_values)
        max_tissue_length = A - B

        C = np.max(sorted_x_values_x)
        D = np.min(sorted_x_values_x)
        max_tissue_width = C - D

        # Define percentages (5% to 90%) for sectional measurement
        percentages = np.arange(5, 95, 5)
        df = pd.DataFrame(columns=['Image', 'Percentage', 'Length', 'Width'])

        # Measure width and length at each defined percentage
        for percentage in percentages:
            # Target Y and X positions at percentage points
            length_percentage = np.round((percentage / 100) * max_tissue_length, 1)
            width_percentage = np.round((percentage / 100) * max_tissue_width, 1)
            width_percentage += np.min(sorted_x_values_x)

            # Find index of closest point to that percentage
            index = np.searchsorted(sorted_y_values, length_percentage, side='left')
            index_x = np.searchsorted(sorted_x_values_x, width_percentage, side='left')

            # Use nearby points for stability in measurement
            indices = [max(0, index - 2), max(0, index - 1), index,
                       min(len(sorted_y_values) - 1, index + 1),
                       min(len(sorted_y_values) - 1, index + 2)]

            indices_x = [max(0, index_x - 2), max(0, index_x - 1), index_x,
                         min(len(sorted_x_values_x) - 1, index_x + 1),
                         min(len(sorted_x_values_x) - 1, index_x + 2)]

            x_vals = sorted_x_values[indices]
            y_vals = sorted_y_values_x[indices_x]
            min_x, max_x = min(x_vals), max(x_vals)
            min_y, max_y = min(y_vals), max(y_vals)

            width_at_percentage = max_x - min_x
            length_at_percentage = max_y - min_y

            # Plot width and length lines on the interpolated contour
            ax3.plot([min_x, max_x], [length_percentage, length_percentage], 'ro-')
            ax3.plot([width_percentage, width_percentage], [min_y, max_y], 'bo-')

            # Overlay measurement lines on original image
            shift = 256 - max_OG_y
            ax4.imshow(contour_frame[::-1])
            ax4.plot([min_x, max_x], [length_percentage + shift] * 2, 'r-', linewidth=0.4)
            ax4.plot([width_percentage] * 2, [min_y + shift, max_y + shift], 'b-', linewidth=0.4)
            ax4.set_xlim([0, 256])
            ax4.set_ylim([0, 256])

            # Append measurement data
            distances_list.append(pd.DataFrame({
                'Image': [current_file],
                'Percentage': [percentage],
                'Length': [length_at_percentage],
                'Width': [width_at_percentage]
            }))

        # Save the annotated image
        output_filename = f"output__{current_file}.png"
        output_filepath = os.path.join(main_folder, output_filename)
        plt.savefig(output_filepath)
        plt.close()

    else:
        print("!!!!no contours found in image!!!!")

# Save final results to Excel 
distances_df = pd.concat(distances_list, ignore_index=True)
excel_file_path = os.path.join(current_dir, 'distances_data.xlsx')
distances_df.to_excel(excel_file_path, index=False)
print(f'Data has been written to {excel_file_path}')
