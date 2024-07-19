from .Model import DeepLabV3Plus
from utils.Processing import split_data, Process


import json
import random
import datetime
import tensorflow as tf
import os
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

def save_history(history, filename):
    """
    Save the training history to a file.
    
    Args:
    - history: The history object returned by model.fit().
    - filename: The filename to save the history to.
    """
    with open(filename, 'w') as f:
        json.dump(history.history, f)

def save_model_config(config, file_path):
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)
        
def train_masks(train_dataset):
    '''
    This function computes class weights for training the masks using a TensorFlow dataset.

    Args:
        train_dataset: TensorFlow Dataset of tuples (image, mask) for training the masks.

    Returns:
        class_weights: Array of class weights based on the distribution of class labels in the masks.
    '''
    print("Computing class weights...")

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


def train_model(deeplab, train_dataset, val_dataset, date_str,dataset_info, batch_size=16, epochs=100):
    # Define the learning rates and optimizer
    model_name = "DeepLab"
    start_lr = 0.001
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
        deeplab
    ])

    model.compile(
        optimizer=optimizer, 
        loss=loss,
        metrics=['accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                ]
        )

    # Directory for checkpoints
    checkpoint_dir = os.path.join(os.getcwd(), "models/DeepLab/checkpoints")
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

    # Training the model with validation data
    history = model.fit(
        train_dataset.batch(batch_size),
        validation_data=val_dataset.batch(batch_size), 
        epochs=epochs, 
        callbacks=callbacks
    )

    # Define the model name and directory for saving the final mode
    final_model_dir = os.path.join(os.getcwd(),"models/DeepLab", model_name, date_str, "Model_Data")
    
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)

    # Save history
    history_file_name = os.path.join(final_model_dir, "history.json")
    save_history(history, history_file_name)

    # Define model save path and save final model
    model_file_name = f"{model_name}_{date_str}.keras"
    model_save_path = os.path.join(final_model_dir, model_file_name)
    model.save(model_save_path)
    
    # Save model configuration
    config = {
        "model_name": model_name,
        "date": date_str,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": {
            "start_lr": start_lr,
            "end_lr": end_lr,
            "decay_steps": decay_steps,
            "schedule": "PolynomialDecay"
        },
        "optimizer": "Adam",
        "loss_function": "BinaryCrossentropy",
        "metrics": ["accuracy", "precision", "recall"],
        "dataset": dataset_info,
        "callbacks": ["ModelCheckpoint", "EarlyStopping"]
    }
    
    config_file_name = os.path.join(final_model_dir, "config.json")
    save_model_config(config, config_file_name)
    
    print(f"Model and history saved successfully in {final_model_dir}")
    return model, history


if __name__ == "__main__":
    '''Training configuration:
            (this will be used before a script is created or not)
    
    '''
    dateset_fileName = "640_640_4.pkl"
    img_height = 640
    img_width = 640
    img_channels = 4
 
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
 
    batch_size = 16
    epochs = 100
 
    
    print("Processing the dataset...")
    dataset = Process(dateset_fileName)
    print("Splitting")
    train_dataset, val_dataset, test_dataset = split_data(dataset)
    # class_weights = train_masks(train_dataset)
    # print(f"Class weights: {class_weights}")
    print("Training the model...")
    deeplab = DeepLabV3Plus(n_classes=1, img_height=img_height, img_width=img_width, img_channels=img_channels)
    
    dataset_info = {
        "dateset_fileName" : dateset_fileName,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "image_shape": (img_height, img_width, img_channels)  # Update this if image shape is different
    }
    
    model, history = train_model(deeplab, train_dataset, val_dataset, date_str,dataset_info, batch_size=batch_size, epochs=epochs)
    # date_str = "2024-07-16-14-06"
    from utils.Evaluation import evaluate
    evaluate(model_date = date_str, model_name = "DeepLab", num_examples=1)