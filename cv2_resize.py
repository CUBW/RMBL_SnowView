import numpy as np
import rasterio
import glob
import cv2
import csv
import os

from tqdm import tqdm

SOURCE_FILEPATH = "C:/Users/apfox/UCB-O365/Travis Hainsworth - RMBL/2 - Source Data/2019-2020_Data" # update to match where your data is
DEST_FILEPATH = "data/640" # update to match where you want it to go

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
    
    # Calculate the new height based on the desired width and the original aspect ratio
    height, width = snow.shape[:2]
    new_height = int(height * new_width / width)

    # Resize snow
    resized_snow = np.zeros((snow.read().shape[0], new_height, new_width), dtype=snow.read().dtype)
    for i in range(snow.read().shape[0]):
        resized_snow[i] = cv2.resize(snow.read()[i], (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Update metadata
    snow.meta.update({
        "height": new_height,
        "width": new_width,
        "transform": snow.transform * snow.transform.scale(
            (width / new_width),
            (height / new_height)
        )
    })

    # Resample mask
    resized_mask_image = np.zeros((mask.read().shape[0], new_height, new_width), dtype=mask.read().dtype)
    for i in range(mask.read().shape[0]):
        resized_mask_image[i] = cv2.resize(mask.read()[i], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Update metadata
    mask.meta.update({
        "height": new_height,
        "width": new_width,
        "transform": mask.transform * mask.transform.scale(
            (mask.read().shape[2] / new_width),
            (mask.read().shape[1] / new_height)
        )
    })    

    # Save the images
    with rasterio.open(os.path.join(DEST_FILEPATH, f"{location}_{date}_snow.tif"), "w", **snow.meta) as dst:
        dst.write(resized_snow)
    
    with rasterio.open(os.path.join(DEST_FILEPATH, f"{location}_{date}_snowbinary.tif"), "w", **mask.meta) as dst:
        dst.write(resized_mask_image)

    # return size of the images
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