import os
import glob
import json
import datetime
import comet_ml
from comet_ml import Experiment
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ---------------------
#  CUSTOM METRICS & CALLBACKS
# ---------------------
def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice Coefficient in a safe way.
    For binary segmentation, y_true and y_pred are expected to be of shape (batch, H, W, 1).
    If the sum of y_true and y_pred is zero, returns 1.0.
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice = tf.where(tf.equal(denom, 0), 1.0, (2. * intersection + smooth) / (denom + smooth))
    return dice

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss for binary segmentation. 
    Note: dice_loss = 1 - dice_coef.
    """
    return 1 - dice_coef(y_true, y_pred, smooth)

def combined_loss(y_true, y_pred, alpha=0.5):
    """
    Combined loss: a weighted sum of Binary Crossentropy and Dice Loss.
    Alpha controls the contribution of BCE (alpha) vs. Dice loss (1 - alpha).
    """
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return alpha * bce + (1 - alpha) * dl

class LearningRateLogger(tf.keras.callbacks.Callback):
    """Logs the learning rate at the end of each epoch to Comet.ml."""
    def __init__(self, experiment=None):
        super().__init__()
        self.experiment = experiment

    def on_epoch_end(self, epoch, logs=None):
        lr_schedule = self.model.optimizer.learning_rate
        if callable(lr_schedule):
            lr = lr_schedule(self.model.optimizer.iterations).numpy()
        else:
            lr = lr_schedule.numpy()
        if self.experiment is not None:
            self.experiment.log_metric("learning_rate", lr, step=epoch)
        print(f"Epoch {epoch+1}: learning_rate = {lr:.6f}")

class DebugBatchStats(tf.keras.callbacks.Callback):
    """Debug callback to print statistics of each training batch."""
    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            #print(f"Batch {batch}: loss = {logs.get('loss', 'N/A'):.4f}")
            this = 1

# ---------------------
#  U-Net MODEL
# ---------------------
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate,
    BatchNormalization, Activation, Dropout
)
from tensorflow.keras.models import Model

def conv_block(x, n_filters, dropout, l2):
    reg = tf.keras.regularizers.L2(l2)
    conv1 = Conv2D(n_filters, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg)(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(dropout)(conv1)
    conv2 = Conv2D(n_filters, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    return conv2

def encoder_block(inputs, n_filters, dropout, l2):
    conv = conv_block(inputs, n_filters, dropout, l2)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool

def decoder_block(inputs, skip_connection, n_filters, dropout, l2):
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(inputs)
    reg = tf.keras.regularizers.L2(l2)
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_regularizer=reg)(x)
    x = concatenate([x, skip_connection])
    x = conv_block(x, n_filters, dropout, l2)
    return x

def unet_model(n_classes, img_height, img_width, img_channels):
    inputs = Input((img_height, img_width, img_channels))
    conv1, pool1 = encoder_block(inputs, 16, 0, 0.0001)
    conv2, pool2 = encoder_block(pool1, 32, 0, 0.0001)
    conv3, pool3 = encoder_block(pool2, 64, 0, 0.0001)
    conv4, pool4 = encoder_block(pool3, 128, 0, 0.001)
    conv5, pool5 = encoder_block(pool4, 256, 0, 0.001)
    conv6, pool6 = encoder_block(pool5, 512, 0, 0.01)
    bridge = conv_block(pool6, n_filters=1024, dropout=0, l2=0.01)
    up6 = decoder_block(bridge, conv6, 512, 0, 0.01)
    up5 = decoder_block(up6, conv5, 256, 0, 0.001)
    up4 = decoder_block(up5, conv4, 128, 0, 0.001)
    up3 = decoder_block(up4, conv3, 64, 0, 0.0001)
    up2 = decoder_block(up3, conv2, 32, 0, 0.0001)
    up1 = decoder_block(up2, conv1, 16, 0, 0.0001)
    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(up1)
    model = Model(inputs=inputs, outputs=outputs)
    print("Model summary:")
    model.summary()
    return model

# ---------------------
#  DATA LOADING WITH VALID RATIO FILTERING
# ---------------------
def load_npy_pair(img_path, msk_path):
    # Load the image and mask arrays.
    img_array = np.load(img_path)
    msk_array_orig = np.load(msk_path)
    
    # Replace extreme values (-3.4e38) with np.nan in the mask.
    msk_array_orig[msk_array_orig == -3.4e38] = np.nan
    
    # Create a weight mask: valid pixels (0 or 1) get weight 1; invalid (nan) get 0.
    weight_mask = np.ones_like(msk_array_orig, dtype=np.float32)
    invalid_mask = np.isnan(msk_array_orig)
    weight_mask[invalid_mask] = 0.0
    
    # Process mask: clip to [0,1] and replace any nan with 0.
    msk_array = np.clip(msk_array_orig, 0, 1)
    msk_array = np.nan_to_num(msk_array, nan=0)
    
    # Check image shape. We expect images to be either (3, 512, 512) or (4, 512, 512)
    if img_array.shape[0] in [3, 4] and img_array.shape[1:] == (512, 512):
        # Transpose to (512,512,channels)
        img_array = np.transpose(img_array, (1, 2, 0))
        # If there are 4 channels, drop the fourth channel
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
    else:
        print(f"Skipping {img_path} due to unexpected image shape: {img_array.shape}", flush=True)
        return None, None, None
        
    # For the mask, we expect a shape of (1,512,512) so we transpose to (512,512,1)
    if msk_array.shape == (1, 512, 512):
        msk_array = np.transpose(msk_array, (1, 2, 0))
        weight_mask = np.transpose(weight_mask, (1, 2, 0))
    else:
        print(f"Skipping {msk_path} due to unexpected mask shape: {msk_array.shape}", flush=True)
        return None, None, None

    return img_array.astype(np.float32), msk_array.astype(np.float32), weight_mask.astype(np.float32)


def pair_generator(pairs_list, valid_threshold=0.1):
    for pair in pairs_list:
        # If pair has more than 2 elements, take only the first two.
        if len(pair) >= 2:
            img_p, msk_p = pair[0], pair[1]
        else:
            continue
        img_array, msk_array, weight_mask = load_npy_pair(img_p, msk_p)
        if img_array is None:
            continue
        total_pixels = weight_mask.size
        valid_ratio = np.sum(weight_mask) / total_pixels
        if valid_ratio < valid_threshold:
            print(f"Skipping {img_p} due to low valid ratio ({valid_ratio:.4f} < {valid_threshold}).", flush=True)
            continue
        yield (img_array, msk_array, weight_mask)

def make_dataset(pairs_list, batch_size=1, shuffle=False, valid_threshold=0.1):
    ds = tf.data.Dataset.from_generator(
        lambda: pair_generator(pairs_list, valid_threshold=valid_threshold),
        output_types=(tf.float32, tf.float32, tf.float32),
        output_shapes=((512, 512, 3), (512, 512, 1), (512, 512, 1))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=50)
        ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------
#  TRAINING SCRIPT
# ---------------------
def save_history(history, filename):
    with open(filename, 'w') as f:
        json.dump(history.history, f)

def save_model_config(config, filename):
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)

def plot_learning_curves(history):
    epochs_range = range(1, len(history.history['loss']) + 1)
    fig, axs = plt.subplots(3, 2, figsize=(16,12))
    
    # Combined Loss plot
    axs[0,0].plot(epochs_range, history.history['loss'], label='Train Combined Loss')
    if 'val_loss' in history.history:
        axs[0,0].plot(epochs_range, history.history['val_loss'], label='Val Combined Loss')
    axs[0,0].set_title("Combined Loss over Epochs")
    axs[0,0].set_xlabel("Epoch")
    axs[0,0].set_ylabel("Loss")
    axs[0,0].legend()
    
    # Dice Coefficient plot
    axs[0,1].plot(epochs_range, history.history['dice_coef'], label='Train Dice Coef')
    if 'val_dice_coef' in history.history:
        axs[0,1].plot(epochs_range, history.history['val_dice_coef'], label='Val Dice Coef')
    axs[0,1].set_title("Dice Coefficient over Epochs")
    axs[0,1].set_xlabel("Epoch")
    axs[0,1].set_ylabel("Dice Coefficient")
    axs[0,1].legend()
    
    # Dice Loss plot (1 - dice_coef)
    train_dice_loss = [1 - x for x in history.history['dice_coef']]
    axs[1,0].plot(epochs_range, train_dice_loss, label='Train Dice Loss')
    if 'val_dice_coef' in history.history:
        val_dice_loss = [1 - x for x in history.history['val_dice_coef']]
        axs[1,0].plot(epochs_range, val_dice_loss, label='Val Dice Loss')
    axs[1,0].set_title("Dice Loss over Epochs")
    axs[1,0].set_xlabel("Epoch")
    axs[1,0].set_ylabel("Dice Loss")
    axs[1,0].legend()
    
    # Accuracy plot
    if 'accuracy' in history.history:
        axs[1,1].plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        axs[1,1].plot(epochs_range, history.history['val_accuracy'], label='Val Accuracy')
    axs[1,1].set_title("Accuracy over Epochs")
    axs[1,1].set_xlabel("Epoch")
    axs[1,1].set_ylabel("Accuracy")
    axs[1,1].legend()
    
    # Precision plot
    if 'precision' in history.history:
        axs[2,0].plot(epochs_range, history.history['precision'], label='Train Precision')
    if 'val_precision' in history.history:
        axs[2,0].plot(epochs_range, history.history['val_precision'], label='Val Precision')
    axs[2,0].set_title("Precision over Epochs")
    axs[2,0].set_xlabel("Epoch")
    axs[2,0].set_ylabel("Precision")
    axs[2,0].legend()
    
    # Recall plot
    if 'recall' in history.history:
        axs[2,1].plot(epochs_range, history.history['recall'], label='Train Recall')
    if 'val_recall' in history.history:
        axs[2,1].plot(epochs_range, history.history['val_recall'], label='Val Recall')
    axs[2,1].set_title("Recall over Epochs")
    axs[2,1].set_xlabel("Epoch")
    axs[2,1].set_ylabel("Recall")
    axs[2,1].legend()
    
    plt.tight_layout()
    plt.show()

def train_model(unet_model, train_dataset, val_dataset, date_str, dataset_info,
                batch_size=8, epochs=200, experiment=None, instance_name="unet"):
    print("Training with batch size:", batch_size)
    print("Number of epochs:", epochs)
    model_name = "unet"
    num_train_samples = dataset_info.get("num_train_samples", None)
    if num_train_samples is not None:
        steps_per_epoch = int(math.ceil(num_train_samples / (batch_size)))
        print("Steps per epoch (auto):", steps_per_epoch)
    else:
        steps_per_epoch = 2000
        print("Steps per epoch (manual):", steps_per_epoch)


    # Calculate validation steps using the total number of validation samples.
    num_val_samples = dataset_info.get("num_val_samples", None)
    if num_val_samples is not None:
        validation_steps = int(math.ceil(num_val_samples / batch_size))
        print("Validation steps (auto):", validation_steps)
    else:
        validation_steps = None  # Keras can infer this if desired

    decay_steps = steps_per_epoch * epochs
    start_lr = 1e-3
    end_lr = 1e-4
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        start_lr, decay_steps, end_lr, power=0.5
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    
    # Use combined_loss instead of BinaryCrossentropy.
    model = unet_model
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        weighted_metrics=[],
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            dice_coef
        ]
    )
    
    checkpoint_dir = os.path.join(os.getcwd(), "models", "unet", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{{epoch:02d}}.keras")
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',  # Monitoring validation combined loss is standard
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )
    
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # You can choose 'val_dice_coef' if dice is more important
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_logger_cb = LearningRateLogger(experiment=experiment)
    debug_batch_cb = DebugBatchStats()
    
    callbacks = [checkpoint_cb, early_stop_cb, lr_logger_cb, debug_batch_cb]
    
    if experiment is not None:
        comet_cb = experiment.get_callback("keras")
        callbacks.append(comet_cb)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    plot_learning_curves(history)
    
    final_model_dir = os.path.join(os.getcwd(), "models", model_name, "Previous", instance_name + '_' + date_str, "Model_Data")
    os.makedirs(final_model_dir, exist_ok=True)
    history_file_name = os.path.join(final_model_dir, "history.json")
    save_history(history, history_file_name)
    model_file_name = f"{instance_name}.keras"
    model_save_path = os.path.join(final_model_dir, model_file_name)
    model.save(model_save_path)
    
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
        "loss_function": "combined_loss (BCE + Dice Loss)",
        "metrics": ["accuracy", "precision", "recall", "dice_coef"],
        "dataset": dataset_info,
        "callbacks": ["ModelCheckpoint", "EarlyStopping", "LearningRateLogger", "DebugBatchStats"] + (["CometCallback"] if experiment else [])
    }
    config_file_name = os.path.join(final_model_dir, "config.json")
    save_model_config(config, config_file_name)
    
    print(f"\nModel and history saved in: {final_model_dir}")
    return model, history

#--------MISC--------#
def get_site_from_tile(tile_path):
    return os.path.basename(tile_path).split("_")[0]
