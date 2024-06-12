import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
from tqdm import tqdm

from rasterio.plot import show
from rasterio.warp import Resampling, reproject

SOURCE_FILEPATH = "C:/Users/apfox/UCB-O365/Travis Hainsworth - RMBL/2 - Source Data/2019-2020_Data" # update to match where your data is
DEST_FILEPATH = "./2019-20" #update to match where you want it to go

def resample(source, target):
    """
    Resamples the source data to match the size and projection of the target data.

    Parameters:
        source (rasterio.DatasetReader): The source dataset to be resampled.
        target (rasterio.DatasetReader): The target dataset with the desired size and projection.

    Returns:
        numpy.ndarray: The resampled data with the same number of channels as the source dataset.

    """
    resampled_data = np.empty((source.count, target.height, target.width), dtype=np.float32)
    for i in range(source.count):  # Loop over channels
        rasterio.warp.reproject(
            source=source.read(i + 1),
            destination=resampled_data[i],
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=target.transform,
            dst_crs=target.crs,
            resampling=Resampling.cubic_spline       
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

def process_images():
    locations = [
        'DeerCreekTrail', 
        'EastRiverTrail', 
        'Elkton', 
        'JuddFalls', 
        'LupineTrail', 
        'ParadiseBasin', 
        'RustlersGulch', 
        'SlideRockRoad', 
        'SnodgrassTrailhead', 
        'SouthBaldy', 
        'StrandHill', 
        'VirginiaBasin'
        ]

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

    #make destination directory if it doesn't exist
    if not os.path.exists(DEST_FILEPATH):
        os.makedirs(DEST_FILEPATH)

    #open csv
    csv_file = open(os.path.join(DEST_FILEPATH,'2019-20.csv'), mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["location", "date", "snow_path", "snowbinary_path"])


    #get all files
    lfp = os.path.join(SOURCE_FILEPATH, "Imagery")
    tif_files = glob.glob(os.path.join(lfp, '**/*.tif'), recursive=True)


    # loop through all files in the directory
    for file_path in tqdm(tif_files, unit = 'image'):

        #get filename
        filename = os.path.basename(file_path)
        tags = filename.split('_')

        #exctract location from filename
        location = tags[0]

        #extract date from filename
        date = f'{tags[1]}_{tags[2]}_{tags[3]}'

        #Skip bad files
        if f'{location}_{date}_snow.tif' in bad_files:
            continue

        #open the images for the selected location and date
        snow_name, snowbinary_name = get_filename(location, date)
        snow_og = rasterio.open(os.path.join(SOURCE_FILEPATH, "Imagery", location, snow_name))
        snow_meta = snow_og.meta

        snowbinary_og = rasterio.open(os.path.join(SOURCE_FILEPATH, "Snow_Mask", location, snowbinary_name))
        snowbinary_meta = snowbinary_og.meta

        #resample snow data to match the binary snow data
        snow_resampled = np.round(resample(snow_og, snowbinary_og))

        #generate shape of snowbinary as mask
        mask = create_mask(snowbinary_og.read(1))

        #trim snow data to match the mask
        snow_trimmed = np.where(mask, snow_resampled, np.nan)

        #fix metadata
        snow_meta = snowbinary_meta.copy()
        snow_meta['count'] = 3

        #store results
        #write snow_trimmed to a new file
        with rasterio.open(os.path.join(DEST_FILEPATH, snow_name), 'w', **snow_meta) as dst:
            for i in range(snow_meta['count']):
                dst.write(snow_trimmed[i], i + 1)

        #write snowbinary to a new file
        with rasterio.open(os.path.join(DEST_FILEPATH, snowbinary_name), 'w', **snowbinary_meta) as dst:
            for i in range(snowbinary_meta['count']):
                dst.write(snowbinary_og.read()[i], i + 1)

        #store filepaths in csv
        writer.writerow([location,date,os.path.join(DEST_FILEPATH, snow_name),os.path.join(DEST_FILEPATH, snowbinary_name)])

    #close csv
    csv_file.close()

if __name__ == "__main__":
    print("Processing images...")
    process_images()
    print("Done processing images.")