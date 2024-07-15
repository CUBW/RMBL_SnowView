import numpy as np
import rasterio
import glob
import cv2
import csv
import os

from tqdm import tqdm
from rasterio.plot import show
from rasterio.warp import Resampling, reproject


SOURCE_FILEPATH = "C:/Users/apfox/UCB-O365/Travis Hainsworth - RMBL/2 - Source Data/2019-2020_Data" # update to match where your data is
DEST_FILEPATH = "data/640" # update to match where you want it to go

def resample(source, target):
    """
    Resamples the source data to match the size and projection of the target data.

    Parameters:
        source (rasterio.DatasetReader): The source dataset to be resampled.
        target (rasterio.DatasetReader): The target dataset with the desired size and projection.

    Returns:
        numpy.ndarray: The resampled data with the same number of channels as the source dataset.

    """
    resampled_data = np.full((source.count, target.height, target.width), np.nan, dtype=np.float32)
    for i in range(source.count):  # Loop over channels
        rasterio.warp.reproject(
            source=source.read(i + 1),
            destination=resampled_data[i],
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=target.transform,
            dst_crs=target.crs,
            resampling=Resampling.nearest,
            src_nodata=None,
            dst_nodata=np.nan
        )
    return resampled_data

def create_mask(data):
    """
    Creates a mask based on the given data.

    Args:
        data (numpy.ndarray): The input data.

    Returns:
        numpy.ndarray: The mask where values are True if they are greater than or equal to -3.3e+38, and False otherwise.
    """
    return data >= 0

def process_image(location, date, new_width=640):
    """
    Process the snow and mask images for a given location and date.
    
    Args:
        location (str): The location of the images.
        date (str): The date of the images.
        new_width (int, optional): The desired width of the resized images. Defaults to 640.
    
    Returns:
        tuple: A tuple containing the new width and height of the resized images.
    """
    # Read the snow image
    snow = rasterio.open(os.path.join(SOURCE_FILEPATH, "Imagery", location, f"{location}_{date}_snow.tif"))
    
    # Read the mask image
    mask = rasterio.open(os.path.join(SOURCE_FILEPATH, "Snow_Mask", location, f"{location}_{date}_snowbinary.tif"))
    
    # resample mask to image aspect ratio
    resampled_mask_data = resample(mask, snow)
    resampled_mask_meta = snow.meta.copy()
    resampled_mask_meta['count'] = 1

    # trim image to mask
    mask_shape = create_mask(resampled_mask_data[0])
    trimmed_snow_data = np.where(mask_shape, snow.read(), np.nan)
   

    # resize image and mask to ___x640
    new_height = int(new_width * (snow.height / snow.width)) # preserve aspect ratio

    resized_snow_data = np.full((snow.count, new_height, new_width), np.nan, dtype=np.float32)
    for i in range(snow.count): 
        resized_snow_data[i] = cv2.resize(trimmed_snow_data[i], (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    resized_mask_data = np.full((1, new_height, new_width), np.nan, dtype=np.float32)
    for i in range(mask.count):
        resized_mask_data[i] = cv2.resize(resampled_mask_data[i], (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    del trimmed_snow_data
    del resampled_mask_data

    # fix metadatas
    resized_snow_meta = snow.meta.copy()
    resized_snow_meta['width'] = new_width
    resized_snow_meta['height'] = new_height
    resized_snow_meta['transform'] = snow.transform * snow.transform.scale(
            (snow.width / new_width),
            (snow.height / new_height)
        )
    
    resized_snow_binary_meta = resampled_mask_meta.copy()
    resized_snow_binary_meta['width'] = new_width
    resized_snow_binary_meta['height'] = new_height
    resized_snow_binary_meta['transform'] = snow.transform * snow.transform.scale(
            (snow.width / new_width),
            (snow.height / new_height)
        )

    # save image and mask
    with rasterio.open(os.path.join(DEST_FILEPATH, f"{location}_{date}_snow.tif"), 'w', **resized_snow_meta) as dst:
        for i in range(resized_snow_meta['count']):
            dst.write(resized_snow_data[i], i + 1)
    
    with rasterio.open(os.path.join(DEST_FILEPATH, f"{location}_{date}_snowbinary.tif"), 'w', **resized_snow_binary_meta) as dst:
        for i in range(resized_snow_binary_meta['count']):
            dst.write(resized_mask_data[i], i + 1)
        
    return (new_width, new_height)

def process_images():
    """
    Process images and generate a CSV file with relevant information.

    This function processes a list of TIFF image files, excluding any files listed in the `bad_files` list.
    It creates a destination directory if it doesn't exist and opens a CSV file to write the processed image information.
    The function then loops through all the TIFF files, extracts location and date information from the filenames,
    skips any files listed in `bad_files`, processes the image, and writes the relevant information to the CSV file.

    Args:
        None

    Returns:
        None
    """
    
    bad_files = [
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
        ]

    # make destination directory if it doesn't exist
    if not os.path.exists(DEST_FILEPATH):
        os.makedirs(DEST_FILEPATH)
    
    # open csv
    csv_file = open(os.path.join(DEST_FILEPATH,'640.csv'), mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["location", "date", "snow_path", "snowbinary_path", "size"])

    # get all files
    tif_files = glob.glob(os.path.join(SOURCE_FILEPATH, "Imagery", '**/*.tif'), recursive=True)

    # loop through all files in the directory
    for file_path in tqdm(tif_files, unit='image'):
        # get filename
        filename = os.path.basename(file_path)
        tags = filename.split('_')

        # exctract location from filename
        location = tags[0]

        # extract date from filename
        date = f'{tags[1]}_{tags[2]}_{tags[3]}'

        # Skip bad files
        if f'{location}_{date}_snow.tif' in bad_files:
            continue

        # process image
        size = process_image(location, date)

        # write to csv
        writer.writerow([location, date, os.path.join(DEST_FILEPATH, f"{location}_{date}_snow.tif"), os.path.join(DEST_FILEPATH,f"{location}_{date}_snowbinary.tif"), size])

if __name__=='__main__':
    print("Processing images...")
    process_images()
    print("Done processing images.")