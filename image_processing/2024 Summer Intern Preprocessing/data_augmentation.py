import os
import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm
# Define paths
input_dir = 'data/640_Squared'
output_dir = 'data/640_Squared_Augmented'
csv_file = os.path.join(input_dir, '640_Squared.csv')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Define the transformations
def apply_transformations(image):
    transformations = []
    for angle in [0, 90, 180, 270]:
        rotated = image.rotate(angle)
        transformations.append(rotated)
        transformations.append(rotated.transpose(Image.FLIP_LEFT_RIGHT))
        transformations.append(rotated.transpose(Image.FLIP_TOP_BOTTOM))
    return transformations

# Process each row in the CSV
for index, row in tqdm(df.iterrows(), unit="image mask pair"):
    image_path = row['snow_path']
    mask_path = row['snowbinary_path']

    # Load images
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Apply transformations
    transformed_images = apply_transformations(image)
    transformed_masks = apply_transformations(mask)

    # Save augmented images and masks
    base_image_name = os.path.splitext(os.path.basename(row['snow_path']))[0]
    base_mask_name = os.path.splitext(os.path.basename(row['snowbinary_path']))[0]

    for i, (transformed_image, transformed_mask) in enumerate(zip(transformed_images, transformed_masks)):
        transformed_image.save(os.path.join(output_dir, f"{base_image_name}_aug_{i}.tif"))
        transformed_mask.save(os.path.join(output_dir, f"{base_mask_name}_aug_{i}.tif"))
