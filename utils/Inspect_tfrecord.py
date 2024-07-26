import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', '512_Splits_4_TFRecord')


# Function to parse a TFRecord example
def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_png(example['image'])
    mask = tf.io.decode_png(example['mask'])
    return image, mask

# Function to visualize images and masks
def visualize_tfrecord(tfrecord_filename, num_examples=10):
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    dataset = dataset.map(parse_tfrecord_fn)
    
    # Convert the dataset to a list to shuffle and select random examples
    dataset_list = list(dataset)
    random.shuffle(dataset_list)
    
    plt.figure(figsize=(10, 10))
    for i, (image, mask) in enumerate(dataset_list[:num_examples]):
        plt.subplot(num_examples, 2, 2 * i + 1)
        plt.imshow(image.numpy())
        plt.title('Image')
        plt.axis('off')
        
        plt.subplot(num_examples, 2, 2 * i + 2)
        plt.imshow(mask.numpy(), cmap='gray')
        plt.title('Mask')
        plt.axis('off')
    
    plt.show()

# Path to the TFRecord file
tfrecord_filename = os.path.join(OUTPUT_DIR, 'train_0.tfrecord')

# Visualize the TFRecord data
visualize_tfrecord(tfrecord_filename)