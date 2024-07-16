
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import random
import cv2
import os
import pickle
IMG_HEIGHT = 640
IMG_WIDTH = 640
IMG_CHANNELS = 3

# Dynamically construct the absolute path to the data file
FILEPATH = os.path.join(os.path.dirname(__file__), '..', 'data', '640_640_4.pkl')





def parse_img_mask(image_path, mask_path):
    # Read image and mask using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None

    # Resize and normalize image and mask
    if image is not None:
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
    if mask is not None:
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT)) / 255.0

    return image, mask

    




def create_dataset(filepath):
    # open pkl file
    with open(filepath, 'rb') as f:
        images, masks = pickle.load(f)
    
    dataset = []
    for image, mask in zip(images, masks):
        dataset.append((image, mask))
    
    return dataset

def list_to_tf_dataset(data_list):
    '''
    Converts a list-based dataset to a TensorFlow Dataset.
    '''
    images, masks = zip(*data_list)
    tf_dataset = tf.data.Dataset.from_tensor_slices((list(images), list(masks)))
    return tf_dataset


def split_data(dataset, train_size=0.8, val_size=0.1, test_size=0.1):
    '''
    Split the data into training, validation, and test sets.

    Args:
        dataset: The dataset to be split as a list of tuples (image, mask).
        train_size: The proportion of data to include in the training set.
        val_size: The proportion of data to include in the validation set.
        test_size: The proportion of data to include in the test set.

    Returns:
        train_dataset: List of tuples for the training set.
        val_dataset: List of tuples for the validation set.
        test_dataset: List of tuples for the test set.
    '''

    # Shuffle dataset
    random.shuffle(dataset)
    
    length = len(dataset)
    
    # Determine the sizes of training, validation, and test sets
    train_size = int(length * train_size)
    val_size = int(length * val_size)
    test_size = int(length * test_size)
    
    # print("Size of training set: ", train_size)
    # print("Size of validation set: ", val_size)
    # print("Size of test set: ", test_size)
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size+val_size]
    test_dataset = dataset[train_size+val_size:train_size+val_size+test_size]
    
    print("Data split successfully")
    return list_to_tf_dataset(train_dataset), list_to_tf_dataset(val_dataset), list_to_tf_dataset(test_dataset)




def train_masks(train_dataset):
    print("Computing class weights...")
    '''
    This function computes class weights for training the masks using a TensorFlow dataset.

    Args:
        train_dataset: TensorFlow Dataset of tuples (image, mask) for training the masks.

    Returns:
        class_weights: Array of class weights based on the distribution of class labels in the masks.
    '''
    class_labels = []  # List to store all class labels from masks

    # Extract class labels from masks in the dataset
    for image, mask in train_dataset:
        flat_mask = tf.reshape(mask, [-1])  # Flatten the mask
        unique_labels = tf.unique(flat_mask)[0].numpy()  # Extract unique labels
        class_labels.extend(unique_labels)
    
    # Calculate class weights based on class imbalance
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(class_labels), y=class_labels)
    print("Class weights computed successfully")
    return class_weights



def Process():
    '''
    This function creates a dataset from the images and masks stored in the specified directory.
    It also inspects the dataset by displaying the shape of the first few images and masks.
    
    Returns: 
        dataset: List of tuples (image, mask) created from the images and masks
            Returned and then split where called after in above functions
    
    
    '''
    dataset = create_dataset(FILEPATH)
    print("Dataset created successfully with {} images".format(len(dataset)))

    # Inspect the first few images and masks in the dataset
    print("\nInspecting the first few images and masks:")
    for image, mask in dataset[:3]:  # Take the first 3 images and masks
        if image is not None:
            print("Image Shape:", image.shape)
        if mask is not None:
            print("Mask Shape:", mask.shape)

    return dataset



if __name__ == "__main__":
    import os
    print("looking at: ", FILEPATH)
    Process()
    
    