from .Model import unet_model
from utils.recordProcessing import create_datasets, select_files
from utils.Evaluation import save_model_config
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

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
    

def train_model(unet_model, train_dataset, val_dataset, date_str, dataset_info, batch_size=20, epochs=60):
    print("batch size: ", batch_size)
    print("num of epochs: ", epochs)
    model_name = "unet"
    # Define the learning rates and optimizer
    start_lr = 0.01
    end_lr = 1e-4
    # look in dataset info for the number of samples in the training dataset
    dataset_length = dataset_info["num_train_samples"]
    decay_steps = dataset_length * 400

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

    model.compile(
        optimizer=optimizer, 
        loss=loss,
        metrics=['accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.MeanIoU(num_classes=2),
                ]
        )

    # Directory for checkpoints
    checkpoint_dir = os.path.join(os.getcwd(), "models/unet/checkpoints")
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
    
    # Calculate steps per epoch
    steps_per_epoch = dataset_info["num_train_samples"] // batch_size
    validation_steps = dataset_info["num_val_samples"] // batch_size

    # Training the model with validation data
    history = model.fit(
        train_dataset,
        validation_data=val_dataset, 
        epochs=epochs, 
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    # Define the model name and directory for saving the final model
    final_model_dir = os.path.join(os.getcwd(),"models",model_name, "Previous", date_str, "Model_Data")
    
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
    
    dateset_DIRName = "512_Splits_4_TFRecord"
    img_height = 512
    img_width = 512
    img_channels = 4
 
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
 
    batch_size = 20
    epochs = 20
    buffer_size = 1000
    
    train_tfrecord_files, test_tfrecord_files, val_tfrecord_files = select_files()
    train_dataset, val_dataset, test_dataset, lengths = create_datasets(train_tfrecord_files, test_tfrecord_files, val_tfrecord_files, batch_size, buffer_size)
    # class_weights = train_masks(train_dataset)
    # print(f"Class weights: {class_weights}")
    print("Training the model...")
    unet_model = unet_model(n_classes=1, img_height=512, img_width=512, img_channels=4)
    dataset_info = {
        "dateset_fileName" : dateset_DIRName,
        "num_train_samples": lengths[0],
        "num_test_samples": lengths[1],
        "num_val_samples": lengths[2],
        "image_shape": (img_height, img_width, img_channels)  
    }
    
    
    model, history = train_model(unet_model, train_dataset, val_dataset, date_str, dataset_info, batch_size=batch_size, epochs=epochs)
    # date_str = "2024-07-16-16-28"
    from utils.Evaluation import evaluate
    evaluate(model_date =date_str, model_name="unet", num_examples=1)
