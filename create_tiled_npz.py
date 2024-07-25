import rasterio
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from rasterio.warp import Resampling
from rasterio import windows
from itertools import product
from rasterio.windows import from_bounds
from concurrent.futures import ThreadPoolExecutor


SOURCE = 'C:/Users/apfox/UCB-O365/Travis Hainsworth - RMBL/2 - Source Data/2019-2020_Data'
OUTPUT_DIRECTORY = 'output_4_channel/'
NUM_CHANNELS = 4 
NUM_THREADS = 10

BAD_FILES = [
        "VirginiaBasin_2019_07_10_snow.tif",
        "VirginiaBasin_2019_07_17_snow.tif",
        "VirginiaBasin_2019_07_24_snow.tif",
        "VirginiaBasin_2019_07_30_snow.tif",
        "VirginiaBasin_2020_08_01_snow.tif",
        "VirginiaBasin_2020_08_08_snow.tif",
        "ParadiseBasin_2020_06_12_snow.tif",
        "ParadiseBasin_2019_08_02_snow.tif",
        "EastRiverTrail_2020_05_26_snow.tif",
        "DeerCreekTrail_2019_05_11_snow.tif",
        "EastRiverTrail_2020_05_05_snow.tif",
        "SouthBaldy_2019_08_02_snow.tif", # errors start here
        "StrandHill_2019_05_16_snow.tif",
        "VirginiaBasin_2020_06_22_snow.tif"
        ]

def get_filename(location, date):
    """
    Generate the filenames for the snow and binary snow images.

    Args:
        location (str): The location of the image.
        date (str): The date of the image.

    Returns:
        tuple: A tuple containing the filenames for the snow and binary snow images.
    """
    return (f"{location}_{date}_snow.tif", f"{location}_{date}_snowbinary.tif")

def resample(source, target_meta):
    """
    Resamples the source data to match the size and projection of the target data.

    Parameters:
        source (rasterio.DatasetReader): The source dataset to be resampled.
        target (rasterio.DatasetReader): The target metadata with the desired size and projection.

    Returns:
        numpy.ndarray: The resampled data with the same number of channels as the source dataset.

    """
    resampled_data = np.empty((source.count, target_meta['height'], target_meta['width']), dtype=np.float32)
    for i in range(source.count):  # Loop over channels
        rasterio.warp.reproject(
            source=source.read(i + 1),
            destination=resampled_data[i],
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=target_meta['transform'],
            dst_crs=target_meta['crs'],
            resampling=Resampling.cubic_spline       
        )
    return resampled_data

def trim_larger_to_smaller(large_src, small_src):
    """
    Trim the larger raster image to the bounds of the smaller raster image.

    Parameters:
        large_src (rasterio.io.DatasetReader): An already opened rasterio dataset for the larger image.
        small_src (rasterio.io.DatasetReader): An already opened rasterio dataset for the smaller image.

    Returns:
        trimmed_data (numpy.ndarray): The trimmed data from the larger image.
        trimmed_meta (dict): The updated metadata for the trimmed GeoTIFF.
    """
    
    # Read metadata and bounds of the smaller image
    small_bounds = small_src.bounds

    # Calculate the window to read from the larger GeoTIFF
    window = from_bounds(
        small_bounds.left, small_bounds.bottom,
        small_bounds.right, small_bounds.top,
        transform=large_src.transform
    )

    # Read the data from the larger GeoTIFF using the calculated window
    trimmed_data = large_src.read(window=window)

    # Update metadata for the trimmed GeoTIFF
    trimmed_meta = large_src.meta.copy()
    trimmed_meta.update({
        "height": round(window.height),
        "width": round(window.width),
        "transform": rasterio.windows.transform(window, large_src.transform)
    })

    return trimmed_data, trimmed_meta

def trim_and_resample(snow_path, mask_path, num_channels=3):
    """
    Trims and resamples the snow image and mask to match each other.

    Parameters:
        snow_path (str): The file path of the snow image.
        mask_path (str): The file path of the mask image.
        num_channels (int, optional): The number of channels in the output image. Defaults to 3.

    Returns:
        tuple: A tuple containing the trimmed snow image, resampled mask image, and metadata.

    Raises:
        None

    """

    # open the images for the selected location and date
    snow_og_src = rasterio.open(snow_path)
    snow_meta = snow_og_src.meta

    snowbinary_og_src = rasterio.open(mask_path)
    snowbinary_meta = snowbinary_og_src.meta

    # trim bigger image to match smaller image with raster windows
    trimmed_snow_image, trimmed_snow_meta = trim_larger_to_smaller(snow_og_src, snowbinary_og_src)

    # resample mask to match snow image
    resampled_snowbinary = resample(snowbinary_og_src, trimmed_snow_meta)

    # create shape of mask
    shape = resampled_snowbinary >= 0 

    # trim snow image to match mask shape
    trimmed_snow_image = np.where(shape, trimmed_snow_image, 0)

    # add 4th alpha channel
    if num_channels == 4:
        alpha = np.where(shape, 255, 0)
        trimmed_snow_image = np.append(trimmed_snow_image, alpha, axis=0)
        trimmed_snow_meta['count'] = 4

    return trimmed_snow_image.astype(np.uint8), resampled_snowbinary.astype(np.uint8), trimmed_snow_meta

def pad_for_tiles(img, mask, meta, tilesize=512):
    """
    Pad the image and mask to be divisible by the tilesize.

    Parameters:
        img (numpy.ndarray): The image to be padded.
        mask (numpy.ndarray): The mask to be padded.
        meta (dict): The metadata for the image and mask.
        tilesize (int): The size of the tiles to be generated.

    Returns:
        img (numpy.ndarray): The padded image.
        mask (numpy.ndarray): The padded mask.
        meta (dict): The updated metadata for the padded image and mask.
    """
    # Calculate the padding required to make the image dimensions divisible by the tilesize
    pad_h = tilesize - (img.shape[1] % tilesize)
    pad_w = tilesize - (img.shape[2] % tilesize)

    # Pad the image and mask
    img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    mask = np.pad(mask, ((0,0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    # Update metadata for the padded image
    meta.update({
        "height": img.shape[1],
        "width": img.shape[2],
        "transform": rasterio.windows.transform(rasterio.windows.Window(0, 0, img.shape[2], img.shape[1]), meta['transform'])
    })

    return img, mask, meta

def tile_windows(meta, tilesize):
    ncols, nrows = meta['width'], meta['height']
    offsets = product(range(0, ncols, tilesize), range(0, nrows, tilesize))
    big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
    for col_off, row_off in offsets:
        window = windows.Window(col_off=col_off, row_off=row_off, width=tilesize, height=tilesize).intersection(big_window)
        yield window

def generate_tiles(image, mask, meta, tilesize = 512):
    image_tiles = []
    mask_tiles = []
    for window in tile_windows(meta, tilesize):

        image_window = image[:, window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width]
        mask_window = mask[:, window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width]

        # if image is 50% empty, skip
        if np.count_nonzero(image_window) < (0.5 * image_window.size):
            continue

        image_tiles.append(image_window)
        mask_tiles.append(mask_window)

    return image_tiles, mask_tiles

def store_tiles(image_tiles, mask_tiles, output_dir, location, date):
    np.savez_compressed(os.path.join(output_dir, f"{location}_{date}_tiles_images.npz"), *image_tiles)
    np.savez_compressed(os.path.join(output_dir, f"{location}_{date}_tiles_masks.npz"), *mask_tiles)

def process_file(file_path, output_dir = OUTPUT_DIRECTORY, store = True):
    """
    Process a file by extracting location and date from the filename,
    trimming and resampling the snow image, generating tiles, and storing
    the tiles if specified.

    Args:
        file_path (str): The path to the file to be processed.
        output_dir (str, optional): The directory to store the tiles. Defaults to OUTPUT_DIRECTORY.
        store (bool, optional): Whether to store the tiles. Defaults to True.
    """

    #get filename
    filename = os.path.basename(file_path)
    tags = filename.split('_')

    #exctract location from filename
    location = tags[0]

    #extract date from filename
    date = f'{tags[1]}_{tags[2]}_{tags[3]}'

    #Skip bad files
    if f'{location}_{date}_snow.tif' in BAD_FILES:
        return

    snow, mask, snow_meta = trim_and_resample(
        os.path.join(SOURCE, 'Imagery', location, f'{location}_{date}_snow.tif'), 
        os.path.join(SOURCE, 'Snow_Mask', location, f'{location}_{date}_snowbinary.tif')
    )

    snow, mask, snow_meta = pad_for_tiles(snow, mask, snow_meta)

    snow_tiles, mask_tiles = generate_tiles(snow, mask, snow_meta)

    if store:
        store_tiles(snow_tiles, mask_tiles, output_dir, location, date)

if __name__ == "__main__":
    # get all tif files from Imagery
    lfp = os.path.join(SOURCE, "Imagery")
    tif_files = glob.glob(os.path.join(lfp, '**/*.tif'), recursive=True)

    # create output directory
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # parallelization
    with ThreadPoolExecutor(NUM_THREADS) as executor:    
        list(tqdm(executor.map(process_file, tif_files), total=len(tif_files)-len(BAD_FILES), unit='images'))

