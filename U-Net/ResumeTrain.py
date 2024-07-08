import numpy as np
import tensorflow as tf
import os
import datetime
from Model import unet_model  
from Train import split_data, Process, save_history
    
def resume_training(model_checkpoint_path, model_name,epochs, start_lr=0.0001, end_lr=1e-6, remaining_epochs=100, dataset = None):
    '''
        This function is able to be used by all different models to resume training from a saved checkpoint.
        
        args:
        - model_checkpoint_path: Path to the checkpoint to resume training from.
        - model_name: Name of the model. ex "U-Net"
        - start_lr: The starting learning rate for the optimizer.
        - end_lr: The ending learning rate for the optimizer.
        - remaining_epochs: Number of epochs to continue training (from the current checkpoint).
            
        Returns:
        - history: The training history of the model.
        - model: The model after finishing training.
    
    '''
    if dataset is None:
        dataset = Process()
    train_dataset, val_dataset, test_dataset = split_data(dataset)
    # Directory for checkpoints
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{{epoch:02d}}.keras")
    
    model = tf.keras.models.load_model(model_checkpoint_path)
    
    
    decay_steps = len(train_dataset) * remaining_epochs
    
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        start_lr,
        decay_steps,
        end_lr,
        power=0.5)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    
    loss = tf.keras.losses.BinaryCrossentropy()
    
    
    model.compile(optimizer=optimizer, 
                loss=loss, 
                metrics=['accuracy',
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall(),
                        tf.keras.metrics.MeanIoU(num_classes=1),
                        ]
                )

    callbacks = [tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_freq='epoch',
        )]

    history = model.fit(
            train_dataset.batch(30),
            validation_data=val_dataset.batch(30), 
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
    model_checkpoint_path = "checkpoints/U-Net_epoch_03.keras"
    model_name = "U-Net"
    model, history = resume_training(model_checkpoint_path, model_name, 3)