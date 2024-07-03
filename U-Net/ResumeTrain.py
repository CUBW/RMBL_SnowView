import os
import datetime
import tensorflow as tf
import numpy as np
from Processing import Process
from Train import save_history, split_data, train_masks

def normalize_mask(mask):
    return tf.cast(mask > (1 / 255.), tf.float32)

def resume_train(model_checkpoint_path, train_dataset, val_dataset, class_weights, start_lr=0.0001, end_lr=1e-6, batch_size=30, remaining_epochs=100):
    '''
    Resumes training of the model from a saved checkpoint.
    
    Args:
    - model_checkpoint_path: Path to the checkpoint to resume training from.
    - train_dataset, val_dataset: Datasets for training and validation.
    - class_weights: Class weights for handling class imbalance.
    - batch_size: Size of the batches.
    - remaining_epochs: Number of epochs to continue training (from the current checkpoint).
    
    Returns:
    - history: The training history of the model.
    '''

    # Load the model from the checkpoint
    model = tf.keras.models.load_model(model_checkpoint_path)
    
    # Extract epoch number from the checkpoint path
    epoch_num = int(model_checkpoint_path.split('_')[-1].split('.')[0])
    print(f"Resuming training from epoch {epoch_num}")

    # Define the learning rates and optimizer, ensuring the learning rate starts from start_lr
    decay_steps = len(train_dataset) * remaining_epochs

    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        start_lr,
        decay_steps,
        end_lr,
        power=0.5
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

    # Configure the loss function and recompile the model
    loss = tf.keras.losses.BinaryCrossentropy()

    # Print model summary for clearing any ambiguity on model structure
    model.summary()

    # Building Sequential model and including the unet_model with rescaling similar to the normal training script
    model_with_rescaling = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        model
    ])

    # Attempt without any metrics first
    model_with_rescaling.compile(
        optimizer=optimizer,
        loss=loss
    )

    # Check and set up datasets
    train_batch = next(iter(train_dataset.batch(batch_size)))
    val_batch = next(iter(val_dataset.batch(batch_size)))
    print("Train batch shape:", train_batch[0].shape, train_batch[1].shape)
    print("Val batch shape:", val_batch[0].shape, val_batch[1].shape)

    # Directory for checkpoints
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.keras")

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
    print("Class Weights: ", class_weights)

    # Reset states for metrics
    for metric in model_with_rescaling.metrics:
        print(f"Resetting state for metric: {metric.name}")
        metric.reset_state()

    # Continue training the model from the loaded state
    history = model_with_rescaling.fit(
        train_dataset.batch(batch_size),
        validation_data=val_dataset.batch(batch_size), 
        initial_epoch=epoch_num + 1,
        epochs=epoch_num + remaining_epochs, 
        callbacks=callbacks,
    )

    # Define the model name and directory for saving the final model
    model_name = "U-Net"
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
    model_with_rescaling.save(model_save_path)
    
    print(f"Model and history saved successfully in {final_model_dir}")
    return model_with_rescaling, history

if __name__ == "__main__":
    # Example usage with dummy values for train_dataset and val_dataset
    model_checkpoint_path = "checkpoints/model_epoch_100.keras"
    dataset = Process()
    train_dataset, val_dataset, test_dataset = split_data(dataset)
    class_weights = train_masks(train_dataset)
    resume_train(model_checkpoint_path, train_dataset, val_dataset, class_weights)
