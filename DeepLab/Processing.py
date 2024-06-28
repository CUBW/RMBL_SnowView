import numpy as np
import tensorflow as tf
import os
import cv2

IMG_HEIGHT = 640
IMG_WIDTH = 640
IMG_CHANNELS = 3

FILEPATH =  "../data/640_Squared/" # update to match where you want it to go

def parse_img_mask(image_path, mask_path):
    """
    Parses the image and mask from the given file paths.

    Args:
        image_path (str): The file path of the image.
        mask_path (str): The file path of the mask.

    Returns:
        tuple: A tuple containing the parsed image and mask.
    """
    # Read image and mask using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None

    # Resize and normalize image and mask
    if image is not None:
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
    if mask is not None:
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT)) / 255.0

    return image, mask

def create_dataset(image_paths, mask_paths):
    """
    Create a dataset by parsing image and mask paths.

    Args:
        image_paths (list): List of image file paths.
        mask_paths (list): List of mask file paths.

    Returns:
        list: A list of tuples, where each tuple contains an image and its corresponding mask.
    """
    dataset = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        image, mask = parse_img_mask(os.path.join(FILEPATH, image_path), os.path.join(FILEPATH, mask_path))
        dataset.append((image, mask))
    return dataset

def Process():
    """
    Process the snow images and masks in the specified directory.

    Returns:
        dataset (list): A list of tuples containing the processed images and masks.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    image_files = [file for file in os.listdir(FILEPATH) if file.endswith('snow.tif')]
    mask_files = [file for file in os.listdir(FILEPATH) if file.endswith('snowbinary.tif')]

    image_files.sort()  # Sort image files
    mask_files.sort()  # Sort mask files
    
    print(len(image_files))
    print("Creating dataset...")
    dataset = create_dataset(image_files, mask_files)
    print("Dataset created successfully with {} images".format(len(dataset)))

    # Inspect the first few images and masks in the dataset
    print("\nInspecting the first few images and masks:")
    for image, mask in dataset[:3]:  # Take the first 3 images and masks
        if image is not None:
            print("Image Shape:", image.shape)
        if mask is not None:
            print("Mask Shape:", mask.shape)

    return dataset

if __name__ == "__main__":
#    print(os.listdir(FILEPATH))
    Process()