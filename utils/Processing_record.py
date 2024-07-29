import tensorflow as tf
import os
import matplotlib.pyplot as plt
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'mask': tf.io.FixedLenFeature([], tf.string)
}

def _parse_function(proto):
    # Parse the input tf.train.Example proto using the feature description.
    parsed_features = tf.io.parse_example(proto, feature_description)
    
    # Decode the image and mask
    images = tf.map_fn(lambda x: tf.io.decode_png(x, channels=4), parsed_features['image'], dtype=tf.uint8)
    masks = tf.map_fn(lambda x: tf.io.decode_png(x, channels=1), parsed_features['mask'], dtype=tf.uint8)
    
    return images, masks


def create_dataset(tfrecord_files, batch_size, buffer_size=1000, training=True):
    # Create a dataset from the TFRecord files
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    
    # Batch the raw dataset before parsing
    raw_dataset = raw_dataset.batch(batch_size)
    
    # Parse the serialized data in the TFRecord files
    parsed_dataset = raw_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if training:
        # Shuffle, batch, and prefetch the data for training
        dataset = parsed_dataset.shuffle(buffer_size).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        # Prefetch the data for validation/testing
        dataset = parsed_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset


def select_files(directory):
    # Get the list of TFRecord files in the directory
    train_tfrecord_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.startswith('train')]
    
    test_tfrecord_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.startswith('test')]
    
    val_tfrecord_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.startswith('val')]
    
    return train_tfrecord_files, test_tfrecord_files, val_tfrecord_files

def visualize_dataset(dataset, num_samples=5):
    # Take a batch of data from the dataset
    for images, masks in dataset.take(1):
        # Convert tensors to numpy arrays
        images = images.numpy()
        masks = masks.numpy()
        
        # Plot the images and masks
        for i in range(num_samples):
            plt.figure(figsize=(10, 5))
            
            # Plot the image
            plt.subplot(1, 2, 1)
            plt.imshow(images[i])
            plt.title('Image')
            plt.axis('off')
            
            # Plot the mask
            plt.subplot(1, 2, 2)
            plt.imshow(masks[i].squeeze(), cmap='gray')
            plt.title('Mask')
            plt.axis('off')
            
            plt.show()

if __name__ == '__main__':
    batch_size = 32
    buffer_size = 1000
    
    DIRPATH = os.path.join(os.path.dirname(__file__), '..', 'data', '512_Splits_4_TFRecord')
    
    # Get the list of TFRecord files
    train_tfrecord_files, test_tfrecord_files, val_tfrecord_files = select_files(DIRPATH) 
    
    # Create datasets for training, testing, and validation
    train_dataset = create_dataset(train_tfrecord_files, batch_size, buffer_size)
    test_dataset = create_dataset(test_tfrecord_files, batch_size, buffer_size, training=False)
    val_dataset = create_dataset(val_tfrecord_files, batch_size, buffer_size, training=False)
    
    visualize_dataset(train_dataset)