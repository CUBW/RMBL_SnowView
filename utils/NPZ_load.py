import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm 

def remove_duplicate_channel(image):
    if image.shape[0] == 5 and np.array_equal(image[3], image[4]):
        updated_image = np.delete(image, 4, axis=0)
        return updated_image
    elif image.shape[0] > 5:
        print("image shape above 5")
        print("image shape before", image.shape)
        updated_image = np.delete(image, 4, axis=0)
        print("image shape after", updated_image.shape)
        return updated_image
    return image

# Function to create a TFRecord example
def create_tfrecord_example(image, mask):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_png(image).numpy()])),
        'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_png(mask).numpy()]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def create_npz_to_TFRecord(image_file, mask_file, tfrecord_filename):
    try:
        # Load the image and mask files
        image_data = np.load(os.path.join(FILEPATH, image_file), allow_pickle=False)
        mask_data = np.load(os.path.join(FILEPATH, mask_file), allow_pickle=False)
    except ValueError as e:
        print(f"Error loading {image_file} or {mask_file}: {e}")
        return
    except Exception as e:
        print(f"Unexpected error loading {image_file} or {mask_file}: {e}")
        return
     
    # Assuming each .npz file contains multiple arrays
    image_keys = list(image_data.keys())
    mask_keys = list(mask_data.keys())
    
    # Create a TFRecord writer for each .npz file
    with tf.io.TFRecordWriter(tfrecord_filename) as tfrecord_writer:
        # Iterate over the image and mask arrays
        category, number = image_file.split('_')[:2]
        for image_key, mask_key in tqdm(zip(image_keys, mask_keys), total=len(image_keys), desc=f"Processing {category}_{number}"):
            try:
                # Load the image and mask
                image = image_data[image_key]
                mask = mask_data[mask_key]
               
                # Remove the duplicate 5th channel if present
                image = remove_duplicate_channel(image)
                
                # Transpose the image and mask to (height, width, channels)
                image = image.transpose(1, 2, 0)
                mask = mask.transpose(1, 2, 0)
                
                # Create a TFRecord example
                if(image.shape[2] >= 5):
                    # image somehow made it past and has more than 4 channels
                    image=image[:,:,:4]
                tfrecord_example = create_tfrecord_example(image, mask)
                
                # Write the example to the TFRecord file
                tfrecord_writer.write(tfrecord_example.SerializeToString())
            except Exception as e:
                print(f"Error processing {image_key} or {mask_key} in {image_file}: {e}")
            
    print(f"Created TFRecord file: {tfrecord_filename}")

def npz_to_TFRecord(FILEPATH, OUTPUT_DIR):
    # Check to see if the input directory exists
    if not os.path.exists(FILEPATH):
        raise FileNotFoundError(f"Input directory '{FILEPATH}' not found.")

    # If the output directory does not exist, create it
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Get all files in the directory
    all_files = os.listdir(FILEPATH)

    # Filter out the split_indicies.txt file and take out the masks
    image_files = [f for f in all_files if not (f.endswith('split_indicies.txt') or f.endswith('masks.npz'))]

    '''
        images and masks are combined in the same file
        the words will be take out in this format
        the files appear with the following format:
            category: train, test, val
            the number in between the _#_ : 0, 1, 2, 3, etc
            the type: images.npz, masks.npz does not matter anymore and will be removed for the new file name
        file extension will be .tfrecord
        example 
            category_#_type.npz
        
        Only taking the category and the number
        resulting in:
            category_#.tfrecord
        example:
            train_0.tfrecord
            test_1.tfrecord
            val_2.tfrecord    
        
    '''

    # Iterate over image_files and create TFRecord files with a progress bar
    for idx, image_file in enumerate(tqdm(image_files, desc="Processing files")):
        # Get the category and number from the image file name
        category, number = image_file.split('_')[:2]
        # Create the TFRecord file name
        tfrecord_filename = os.path.join(OUTPUT_DIR, f'{category}_{number}.tfrecord')
        # Create the TFRecord file
        create_npz_to_TFRecord(image_file, image_file.replace('images', 'masks'), tfrecord_filename)
        print(f"Created TFRecord file: {tfrecord_filename}")
        print(f"Processed {idx + 1}/{len(image_files)} files.")

if __name__ == "__main__":
    FILEPATH = os.path.join(os.path.dirname(__file__), '..', 'data', '1024_splits_4_channel')
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', '1024_splits_4_TFRecord')
    npz_to_TFRecord(FILEPATH, OUTPUT_DIR)
