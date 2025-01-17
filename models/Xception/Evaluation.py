from Processing import Process  # Assuming this is where necessary classes are imported

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import json
import cv2
import random

from datetime import datetime
from sklearn.metrics import confusion_matrix


def save_plot(plt, filename, model_name, date_str = datetime.now().strftime("%Y-%m-%d") ):
    """
    Save the plot to a file with a structured directory based on model name and current date.
    
    Args:
    - plt: The plot object to save.
    - filename: The filename template to save the plot to.
    - model_name: The name of the model for directory structuring.

    Result:
    plot is saved in: /model/{model_name}/{current_date}/results/{filename}
    model is saved in: /model/{model_name}/{current_date}/Model_Data/{filename}
    """
    directory = os.path.join(model_name, date_str,"results")
    if not os.path.exists(directory):
        os.makedirs(directory)
    full_filepath = os.path.join(directory, filename)
    plt.savefig(full_filepath)
    plt.close()


def load_history(filename):
    """
    Load the training history from a file.
    
    Args:
    - filename: The filename to load the history from.
    
    Returns:
    A dictionary containing the training history.
    """
    with open(filename, 'r') as f:
        history_dict = json.load(f)
    return history_dict


def evaluate_model(model, history, train_dataset, val_dataset, test_dataset, model_name):
    # Print type and content of history for debugging

    if not isinstance(history, dict):
        print("Error: history is not in the expected dictionary format.")
        return

    try:
        # Accessing the history values
        loss = history['loss']
        val_loss = history['val_loss']
        acc = history['accuracy']
        val_acc = history['val_accuracy']
    except KeyError as e:
        print(f"Error: key {e} not found in history.")
        return
    except IndexError as e:
        print(f"Error: index {e} out of range.")
        return

    epochs = range(1, len(loss) + 1)

    # Plot training & validation loss and accuracy
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label='Training accuracy')
    plt.plot(epochs, val_acc, label='Validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save plot
    save_plot(plt, 'training_validation_loss_accuracy.png', model_name)

    # Show plot on screen (if running in an environment that supports plotting)
    plt.show()

    # # Evaluate on training dataset
    # train_loss = model.evaluate(train_dataset.batch(8))[0]
    # # Evaluate on validation dataset
    # val_loss_eval = model.evaluate(val_dataset.batch(8))[0]
    # # Evaluate on test dataset
    # test_loss = model.evaluate(test_dataset.batch(8))[0]

    # print(f"Train loss: {train_loss}")
    # print(f"Validation loss: {val_loss_eval}")
    # print(f"Test loss: {test_loss}")

    # Generate predictions and confusion matrix
    predictions = model.predict(test_dataset.batch(8))
    test_images_list = list(test_dataset.map(lambda x, y: x))
    test_masks_list = list(test_dataset.map(lambda x, y: y))
    
    test_images = np.concatenate([image_batch.numpy() for image_batch in test_images_list], axis=0)
    test_masks = np.concatenate([mask_batch.numpy() for mask_batch in test_masks_list], axis=0)

    # Assuming test_labels are in one-hot encoded format
    test_labels = np.argmax(test_masks, axis=1)
    pred_labels = np.argmax(predictions, axis=1)

    if len(test_labels) != len(pred_labels):
        print(f"Inconsistent number of samples: true labels - {len(test_labels)}, predictions - {len(pred_labels)}")
        return test_images, test_masks, predictions

    conf_matrix = confusion_matrix(test_labels, pred_labels)
    print("Confusion Matrix:\n", conf_matrix)
    



def visualize(img, mask, pred_image, location=None, date=None):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Ensure img is a numpy array and convert depth to uint8
    img = np.array(img)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)  # Rescale if needed and convert to uint8
    
    if pred_image.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    # Display original image
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # Display mask
    axs[0, 1].imshow(mask, cmap='gray')
    axs[0, 1].set_title('Mask')
    axs[0, 1].axis('off')

    # Display predicted image
    axs[1, 0].imshow(pred_image, cmap='gray')
    axs[1, 0].set_title('Predicted Image')
    axs[1, 0].axis('off')

    # If needed, leave the fourth subplot empty
    axs[1, 1].axis('off')

    # Add location and date to the title
    title = ""
    if location:
        title += f'Location: {location}'
    if date:
        if title:
            title += f', Date: {date}'
        else:
            title += f'Date: {date}'
    if title:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()

    return plt



def visualize_predictions(dataset, model, location=None, date=None, num_examples=3, fileDir=None):
    # Take a random sample of images from the dataset
    samples = random.sample(list(dataset), num_examples)
    
    # Separate images and masks
    imgs, masks = zip(*samples)
    imgs_array = np.array(imgs)
    
    predictions = model.predict(imgs_array)
    
    
    for i, (img, mask) in enumerate(samples):
        plt = visualize(img, mask, predictions[i], location, date)
        if fileDir:
            save_plot(plt, f'prediction_{i}.png', fileDir)




if __name__ == "__main__":
    from Train import split_data

    # Construct the absolute path for loading the model
    model_path = os.path.abspath(os.path.join("Xception", "2024-07-03-10-58", "Model_Data", "Xception_2024-07-03-10-58.keras"))
    print(f"Loading Model from Path: {model_path}")
    try:
        # Load the saved Xception model
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except ValueError as e:
        print(f"File not found error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")

    model_name = "Xception"

    # Path to the history file
    history_path = os.path.abspath(os.path.join("Xception", "2024-07-03-10-58", "Model_Data", "history.json"))
    
    try:
        history = load_history(history_path)
        print("History loaded and parsed successfully.")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        history = None
    except json.JSONDecodeError as e:
        print(f"Error parsing history JSON: {e}")
        history = None
    except Exception as e:
        print(f"An unexpected error occurred while reading the history: {e}")
        history = None

    # Ensure history is loaded before proceeding
    if history is not None:
        # Load the processed dataset
        dataset = Process()
        train_dataset, val_dataset, test_dataset = split_data(dataset)
        evaluate_model(model, history, train_dataset, val_dataset, test_dataset, model_name)
        visualize_predictions(train_dataset, model , num_examples=1)