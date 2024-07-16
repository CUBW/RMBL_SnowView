import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm

# Define the transformations
def apply_transformations(image):
    transformations = []
    for angle in range(4):
        rotated = np.rot90(image, k=angle)
        transformations.append(rotated)
        transformations.append(np.fliplr(rotated))
        transformations.append(np.flipud(rotated))
    return transformations

if __name__ =="__main__":
    # load in data with pickle
    with open('../data/640_640_4.pkl', 'rb') as f:
        images, masks = pickle.load(f)

    new_images = []
    new_masks = []

    for image, mask in tqdm(zip(images, masks), unit='image mask pair'):
        # Apply transformations
        transformed_images = apply_transformations(image)
        transformed_masks = apply_transformations(mask)

        new_images.extend(transformed_images)
        new_masks.extend(transformed_masks)
    
    # save the new images and masks
    with open('../640_640_4_aug.pkl', 'wb') as f:
        pickle.dump((new_images, new_masks), f)