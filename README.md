Requirements
----------------
Install all required packages with: pip install -r requirements.txt


Steps
------
1. Train the Model: Train your tissue segmentation model using train.py
	- Make sure your training images are placed in data/train/ 
	- Loads and preprocesses training data (resize, normalize).
	- Builds and compiles a CNN model (U-Net)
	- Uses callbacks for model checkpointing and early stopping.
	- Saves the final trained model to models/model.keras.


2. Run Image Processing & Measurements: Using trained model on test images with process_images.py
	- Loads images from data/test/
	- Applies model to generate segmentations
	- Extracts contours and computes:
	- Tissue width at 5% to 95% of the height
	- Tissue length at 5% to 95% of the width
	- Generates and saves overlay plots
	- Exports data to an Excel file in results/


Output
-------
After running the analysis script, you’ll get:
- Visual results: Overlays saved as .png in data/output/
- Quantitative data: Excel file results/distances_data.xlsx
