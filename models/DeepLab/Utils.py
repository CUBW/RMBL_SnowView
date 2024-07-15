import random
import tensorflow as tf

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
    
    print("Size of training set: ", train_size)
    print("Size of validation set: ", val_size)
    print("Size of test set: ", test_size)
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size+val_size]
    test_dataset = dataset[train_size+val_size:train_size+val_size+test_size]
    
    print("Data split successfully")
    return list_to_tf_dataset(train_dataset), list_to_tf_dataset(val_dataset), list_to_tf_dataset(test_dataset)
