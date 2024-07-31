import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import pickle
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

def pad_image(image_array, tile_size):
    height, width = image_array.shape[:2]
    # Calculate new dimensions
    new_height = (height // tile_size + 1) * tile_size
    new_width = (width // tile_size + 1) * tile_size
    
    # Create a padded image
    padded_image = np.zeros((new_height, new_width, image_array.shape[2]), dtype=image_array.dtype)
    padded_image[:height, :width] = image_array
    
    return padded_image

def split_into_tiles(image_array, tile_size):
    height, width = image_array.shape[:2]
    tiles = []
    for x in range(0, width, tile_size):
        for y in range(0, height, tile_size):
            window = Window(x, y, tile_size, tile_size)
            tile = image_array[window.row_off:window.row_off+window.height, window.col_off:window.col_off+window.width]
            tiles.append(tile)
    return tiles

def preprocess_tile(tile):
    # Convert tile to (num_pixels, 3) for RGB if tile has 3 channels
    if tile.ndim == 3 and tile.shape[2] == 3:
        return tile.reshape(-1, 3)
    else:
        raise ValueError("Tile does not have 3 channels for RGB.")

def postprocess_tile(predictions, tile_size):
    # Reshape predictions back to the original tile shape
    return predictions.reshape((tile_size, tile_size))

def combine_tiles(tiles, image_size, tile_size):
    width, height = image_size
    combined_image = np.zeros((height, width), dtype=np.uint8)
    idx = 0
    for x in range(0, width, tile_size):
        for y in range(0, height, tile_size):
            combined_image[y:y+tile_size, x:x+tile_size] = tiles[idx]
            idx += 1
    return combined_image

def main(filename, classifier_pickle_file):
    # Open the TIFF image using rasterio
    with rasterio.open(filename) as src:
        image_array = src.read([1, 2, 3])  # Read RGB bands
        image_array = np.moveaxis(image_array, 0, -1)  # Move bands to last dimension

    # Convert to 8-bit if needed (assuming image array is in [0, 1] range)
    if image_array.max() <= 1:
        image_array = (image_array * 255).astype(np.uint8)
    
    # Pad the image to make its dimensions divisible by the tile size
    tile_size = 1024
    padded_image = pad_image(image_array, tile_size)
    
    # Split the padded image into tiles
    tiles = split_into_tiles(padded_image, tile_size)
    
    # Load the SGDClassifier from pickle file
    with open(classifier_pickle_file, 'rb') as f:
        clf = pickle.load(f)
    
    processed_tiles = []
    
    for tile in tqdm(tiles):
        # Preprocess tile
        tile_data = preprocess_tile(tile)
        
        # Predict using the SGDClassifier
        predictions = clf.predict(tile_data)
        
        # Postprocess tile
        prediction_image = postprocess_tile(predictions, tile_size)
        
        # Append processed tile as image
        processed_tiles.append(prediction_image)
        
    # print unique values in predictions
    print(np.unique(processed_tiles, return_counts=True))
    
    # Combine processed tiles into one image
    predicted_image = combine_tiles(processed_tiles, padded_image.shape[:2], tile_size)
    
    # Crop the predicted image to the original image size
    original_height, original_width = image_array.shape[:2]
    predicted_image_cropped = predicted_image[:original_height, :original_width]
    
    # Convert arrays to PIL images for displaying
    original_image_pil = Image.fromarray(image_array)
    predicted_image_pil = Image.fromarray(predicted_image_cropped, mode='L')

    # Display original and predicted images side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(original_image_pil)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(predicted_image_pil, cmap='gray')
    axs[1].set_title('Predicted Image')
    axs[1].axis('off')
    
    plt.show()
    
if __name__ == "__main__":
    filename = 'E:/RMBL_Snow_Data/2019-2020_Data/Imagery/DeerCreekTrail/DeerCreekTrail_2020_05_05_snow.tif'  # Replace with the path to your TIFF file
    classifier_pickle_file = '3_channel_results/checkpoints/3_feature_sgd_30.pkl'  # Replace with the path to your pickled SGDClassifier
    main(filename, classifier_pickle_file)
