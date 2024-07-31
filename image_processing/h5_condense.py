import os
import glob
import numpy as np
import cv2
import h5py
import time

from tqdm import tqdm
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.segmentation import felzenszwalb, quickshift, slic, watershed
from skimage import filters
from skimage.morphology import disk
from scipy import ndimage as nd

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

SOURCE_DIR = "E:/RMBL_Snow_Data/1024_splits_4_channel"
OUTPUT_DIR = "E:/RMBL_Snow_Data/1024_image_mask"


def process_file(file,lock, output_dir):
    tags = os.path.basename(file).split('_')
    type = tags[1]

    # Open images and masks
    images = np.load(file)
    masks = np.load(file.replace('images', 'masks'))

    images_arrays = images.files
    masks_arrays = masks.files

    batch = np.empty((0, 5, 1024, 1024), dtype=np.uint8)
    batch_size = 20
    

    for i in range(len(images_arrays)):
        image = images[images_arrays[i]].astype(np.uint8)
        
        if image.shape[0]==5:
            # delete 5th dimension
            image = np.delete(image, -1, axis=0)
            
        mask = masks[masks_arrays[i]].astype(np.uint8)
        
        batch = np.concatenate((batch,np.append(image, mask, axis=0)[np.newaxis, :]), axis=0)
        
        if i%batch_size>=batch_size-1:
        # print(f"Processed {batch_size} images in {time.time()-start_time}.")
        # start_time = time.time()
            with lock:
                with h5py.File(os.path.join(output_dir, 'image_mask.h5'), 'a') as hf:
                    if type in hf.keys():
                        # Append
                        hf[type].resize((hf[type].shape[0] + batch.shape[0]), axis=0)
                        hf[type][-batch.shape[0]:] = batch
                    else:
                        # Create dataset
                        hf.create_dataset(type, data=batch, compression="gzip", maxshape=(None, 5, 1024, 1024), chunks=True)
                batch = np.empty((0, 5, 1024, 1024), dtype=np.uint8)
                
    with lock:
        with h5py.File(os.path.join(output_dir, 'image_mask.h5'), 'a') as hf:
            if type in hf.keys():
                # Append
                hf[type].resize((hf[type].shape[0] + batch.shape[0]), axis=0)
                hf[type][-batch.shape[0]:] = batch
            else:
                # Create dataset
                hf.create_dataset(type, data=batch, compression="gzip", maxshape=(None, 5, 1024, 1024), chunks=True)

def main(source_dir, output_dir):
    # create ouput directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all _images.npz files from SOURCE_DIR
    npz_files = glob.glob(os.path.join(source_dir, '*_images.npz'), recursive=True)

    # Create a lock for thread-safe HDF5 file access
    lock = multiprocessing.Manager().Lock()

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(process_file, file, lock, output_dir) for file in npz_files]
        for future in tqdm(as_completed(futures), total=len(futures), unit='file'):
            future.result()  # To raise any exceptions that occurred during execution
            
if __name__ == "__main__":
    main(SOURCE_DIR, OUTPUT_DIR)