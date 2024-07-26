import numpy as np
import os
import matplotlib.pyplot as plt
import random

# Load the .npz file
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', '512_Splits_4_channel_Record')
FILEPATH = os.path.join(os.path.dirname(__file__), '..', 'data', '512_splits_4_channel.nosync')
# Load the .npz file
file_path = os.path.join(FILEPATH, 'train_2_masks.npz')
# print the files at the path
# print(os.listdir(FILEPATH))


data = np.load(file_path)

# Inspect the keys
# Function to display an image with 5 channels
# Function to display an image with 5 channels
def display_5_channel_image(image, key):
    channels, height, width = image.shape
    
    # Display the first three channels as an RGB image
    rgb_image = image[:3].transpose(1, 2, 0)
    plt.figure(figsize=(15, 5))
    plt.suptitle(f'Key: {key}', fontsize=16)
    
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)
    plt.title('RGB Image')
    
    # Display the 4th channel as a grayscale image
    plt.subplot(1, 3, 2)
    plt.imshow(image[3], cmap='gray')
    plt.title('4th Channel')
    
    # Display the 5th channel as a grayscale image
    plt.subplot(1, 3, 3)
    plt.imshow(image[4], cmap='gray')
    plt.title('5th Channel')
    
    plt.show()

# Collect keys for images with 5 channels
keys_with_5_channels = [key for key in data.keys() if data[key].ndim == 3 and data[key].shape[0] == 5]

# Randomly select 5 keys
random_keys = random.sample(keys_with_5_channels, min(10, len(keys_with_5_channels)))

# Display the selected images
for key in random_keys:
    array = data[key]
    print(f"Displaying image for key: {key}, Shape: {array.shape}")
    display_5_channel_image(array, key)


