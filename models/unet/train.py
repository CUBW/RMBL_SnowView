from .Model import unet_model
from utils.Processing import Process, split_data



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
    



def train_model(unet_model, train_dataset, val_dataset, test_dataset, date_str, batch_size=10, epochs=1 ):
    print("batch size: ", batch_size)
    print("num of epochs: ", epochs)
    model_name = "U-Net"
    # Define the learning rates and optimizer
    start_lr = 0.001
    end_lr = 1e-4
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
                        tf.keras.metrics.MeanIoU(num_classes=2),
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
    # if isinstance(class_weights, np.ndarray):
    #     print("Converting class weights to dictionary")
    #     class_weights = {i: weight for i, weight in enumerate(class_weights)}
    
    # Training the model with validation data
    history = model.fit(
        train_dataset.batch(batch_size),
        validation_data=val_dataset.batch(batch_size), 
        epochs=epochs, 
        callbacks=callbacks,
    )

    # Define the model name and directory for saving the final model
    
   
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
    # dataset = Process()
    # train_dataset, val_dataset, test_dataset = split_data(dataset)
    # # class_weights = train_masks(train_dataset)
    # # print(f"Class weights: {class_weights}")
    # print("Training the model...")
    # unet_model = unet_model(n_classes=1, img_height=640, img_width=640, img_channels=4)
    # date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    # model, history = train_model(unet_model, train_dataset, val_dataset, test_dataset, date_str)
    from utils.Evaluation import evaluate
    evaluate(model_date = "2024-07-08-15-02", num_examples=1)
