import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# Finding the average size of the photo
# Directory containing the '_snow.tif' files
directory = '../data/640/'


def resize_and_save_images(directory):
    # Create the output directory if it doesn't exist
    output_directory = os.path.join(directory, "../640_Squared")
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(('.tif')):
            # Open the image
            image_path = os.path.join(directory, filename)
            with Image.open(image_path) as img:
                # Resize the image to have a square aspect ratio of 640x640 pixels
                img = img.resize((640, 640))

                # Save the resized image to the output directory
                output_path = os.path.join(output_directory, filename)
                img.save(output_path)

    print("Images resized and saved to /640_Squared/ directory.")



# Dictionary to store the frequency of aspect ratios
aspect_ratio_freq = {}

# Function to calculate the aspect ratio of an image
def calculate_aspect_ratio(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        return width / height

# Iterate through each file in the directory
image_files = [file for file in os.listdir(directory) if file.endswith(('.tif'))]

for image_file in image_files:
    image_path = os.path.join(directory, image_file)
    aspect_ratio = calculate_aspect_ratio(image_path)
    
    # Increment the frequency count for the aspect ratio
    aspect_ratio_freq[aspect_ratio] = aspect_ratio_freq.get(aspect_ratio, 0) + 1

# Find the aspect ratio with the highest frequency
max_frequency = max(aspect_ratio_freq.values())
max_aspect_ratio = [ar for ar, freq in aspect_ratio_freq.items() if freq == max_frequency][0]

resize_and_save_images(directory)
