import rasterio
import numpy as np
import os
import glob

from tqdm import tqdm
SOURCE_FILEPATH = 'data/640/'
DEST_FILEPATH = 'data/640x1132/'

import rasterio
import numpy as np

def pad_geotiff(input_path, output_path, target_width=640, target_height=1132):
    """
    Pad a GeoTIFF image with np.nan values to achieve the target width and height.

    Parameters:
        input_path (str): The path to the input GeoTIFF file.
        output_path (str): The path to save the padded GeoTIFF file.
        target_width (int): The desired width of the padded image. Default is 640.
        target_height (int): The desired height of the padded image. Default is 1132.
    """
    with rasterio.open(input_path) as src:
        # Read the original data
        data = src.read()
        bands, original_height, original_width = data.shape
        
        # Calculate padding
        pad_height_top = (target_height - original_height) // 2
        pad_height_bottom = target_height - original_height - pad_height_top
        pad_width_left = (target_width - original_width) // 2
        pad_width_right = target_width - original_width - pad_width_left
        
        # Create padded array with np.nan for each band
        padded_data = np.pad(data, ((0, 0), (pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right)), mode='constant', constant_values=np.nan)
        
        # Update metadata
        transform = src.transform
        new_transform = transform * transform.scale((src.width / padded_data.shape[-1]), (src.height / padded_data.shape[-2]))
        
        new_meta = src.meta.copy()
        new_meta.update({
            "driver": "GTiff",
            "height": target_height,
            "width": target_width,
            "transform": new_transform
        })
        
        # Write the new image
        with rasterio.open(output_path, 'w', **new_meta) as dst:
            dst.write(padded_data)

# Directories
os.makedirs(DEST_FILEPATH, exist_ok=True)

# Process each file
for input_path in tqdm(glob.glob(os.path.join(SOURCE_FILEPATH, '*.tif')), unit='image'):
    filename = os.path.basename(input_path)
    output_path = os.path.join(DEST_FILEPATH, filename)
    pad_geotiff(input_path, output_path)
