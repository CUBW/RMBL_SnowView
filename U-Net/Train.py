from Model import unet_model
from Processing import Process



import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json
import random
import datetime

def save_history(history, filename):
    """
    Save the training history to a file.
    
    Args:
    - history: The history object returned by model.fit().
    - filename: The filename to save the history to.
    """
    with open(filename, 'w') as f:
        json.dump(history.history, f)



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
    
    



def train_model(unet_model, train_dataset, val_dataset, test_dataset, class_weights, batch_size=30, epochs=20):
    model_name = "U-Net"
    # Define the learning rates and optimizer
    start_lr = 0.0001
    end_lr = 1e-6
    decay_steps = len(train_dataset) * 400

    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        start_lr,
        decay_steps,
        end_lr,
        power=0.5)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

    # Configure the loss function with the weights
    loss = tf.keras.losses.BinaryCrossentropy()

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        unet_model
    ])

    model.compile(optimizer=optimizer, 
                loss=loss,
                metrics=['accuracy',
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall(),
                        tf.keras.metrics.MeanIoU(num_classes=1),
                        ]
                )

    # Directory for checkpoints
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{{epoch:02d}}.keras")

    # Model checkpointing to save the best model based on validation loss
    callbacks = [tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_freq='epoch',
    )]
    # Convert class_weights to dictionary if it's a numpy array
    if isinstance(class_weights, np.ndarray):
        print("Converting class weights to dictionary")
        class_weights = {i: weight for i, weight in enumerate(class_weights)}
    
    # Training the model with validation data
    history = model.fit(
        train_dataset.batch(batch_size),
        validation_data=val_dataset.batch(batch_size), 
        epochs=epochs, 
        callbacks=callbacks,
    )

    # Define the model name and directory for saving the final model
    
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    final_model_dir = os.path.join(os.getcwd(), model_name, date_str, "Model_Data")
    
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)

    # Save history
    history_file_name = os.path.join(final_model_dir, "history.json")
    save_history(history, history_file_name)

    # Define model save path and save final model
    model_file_name = f"{model_name}_{date_str}.keras"
    model_save_path = os.path.join(final_model_dir, model_file_name)
    model.save(model_save_path)
    
    print(f"Model and history saved successfully in {final_model_dir}")
    return model, history


if __name__ == "__main__":
    from Evalutation import visualize_predictions
    dataset = Process()
    train_dataset, val_dataset, test_dataset = split_data(dataset)
    class_weights = train_masks(train_dataset)
    print(f"Class weights: {class_weights}")
    print("Training the model...")
    unet_model = unet_model(n_classes=1, img_height=640, img_width=640, img_channels=3)
    model, history = train_model(unet_model, train_dataset, val_dataset, test_dataset, class_weights)
    from Evalutation import evaluate_model
    # date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    date_str  = "2024-07-08-15-02"
    model_name = "U-Net" + "_" + date_str 
    save_path = "U-Net/" + date_str + "/results/"
    evaluate_model(model, history, train_dataset,val_dataset, test_dataset, save_path=model_name)
    visualize_predictions(train_dataset, model , num_examples=1, fileDir=save_path)

    
