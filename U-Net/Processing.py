
import numpy as np
from Model import unet_model
import tensorflow as tf
import os

IMG_HEIGHT = 640
IMG_WIDTH = 640
IMG_CHANNELS = 3

FILEPATH =  "../data/640/" # update to match where you want it to go



def parse_img_mask(image_path, mask_path):
    '''
    This function is just meant for loading images and masks, not intended to do any processing
    Assuming iamges are already 640 x 640 
    
    If more processing is needed, it can be added here
    
    '''
    image = tf.io.read_file(image_path)
    image = tf.cast(image, tf.float32)
    
    mask = tf.io.read_file(mask_path)
    return image, mask

def create_datatset(image_paths, mask_paths):
    '''
    This function is meant to create a dataset from the images and masks
    
    '''
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(parse_img_mask)
    return dataset
    
    
    
def Processing():
    image_files = [file for file in os.listdir(FILEPATH) if file.endswith(('.tif'))]
    


if __name__ == "__main__":
    main()