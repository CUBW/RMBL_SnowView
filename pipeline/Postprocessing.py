import rasterio
import matplotlib.pyplot as plt

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