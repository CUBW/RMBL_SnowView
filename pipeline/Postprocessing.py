import rasterio
import matplotlib.pyplot as plt
import numpy as np

def save_image(img, meta, filepath):
    """
    Save the image as a GeoTIFF using the given metadata.

    Args:
        img (numpy.ndarray): The image data to be saved.
        meta (dict): The metadata for the image.
        filepath (str): The path to the output file.

    Returns:
        None
    """
    with rasterio.open(filepath, 'w', **meta) as dst:
        for i in range(meta['count']):
            dst.write(img[i], i + 1)

def display_both(img, mask):
    """
    Display both the original image and the mask side by side.

    Parameters:
    img (numpy.ndarray): The original image.
    mask (numpy.ndarray): The mask image.

    Returns:
    None
    """
    _, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(mask, cmap='gray')
    ax[1].axis('off')
    plt.show()

def combine_tiles(tiles, meta, tile_sizes,  output_filepath):
    """
    Combine the list of tiles into a single image and save it.

    Args:
        tiles (list): A list of numpy arrays representing the tiles.
        meta (dict): The metadata for the image.
        output_filepath (str): The path to the output file.

    Returns:
        None
    """
    # combine the tiles
    height = tile_sizes[0]
    width = tile_sizes[1]

    grid = np.reshape(tiles, (height, width, 1024, 1024))


    large_array = np.block([[grid[i, j] for j in range(29)] for i in range(24)])

    pred = np.expand_dims(large_array, axis=0) # add dimension

    # save the image
    save_image(pred, meta, output_filepath)

    return pred