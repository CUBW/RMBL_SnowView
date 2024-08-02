import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import tensorflow as tf
import json
import cv2
import random

from utils.recordProcessing import select_test_files, create_test_dataset

def save_plot(fig, filename, fileDir):
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)
    full_filepath = os.path.join(fileDir, filename)
    fig.savefig(full_filepath)
    plt.close(fig)

def save_model_config(config, file_path):
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

def load_history(filename):
    with open(filename, 'r') as f:
        history_dict = json.load(f)
    return history_dict

def evaluate_model(model, history, test_dataset, save_path):
    if not isinstance(history, dict):
        print("Error: history is not in the expected dictionary format.")
        return

    try:
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

    fig = plt.figure(figsize=(20, 5))

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

    save_plot(fig, 'training_validation_loss_accuracy.png', save_path)
    plt.show()

    predictions = model.predict(test_dataset)

    test_images_list = list(test_dataset.map(lambda x, y: x))
    test_masks_list = list(test_dataset.map(lambda x, y: y))

    test_images = np.concatenate([image_batch.numpy() for image_batch in test_images_list], axis=0)
    test_masks = np.concatenate([mask_batch.numpy() for mask_batch in test_masks_list], axis=0)

    test_labels = np.argmax(test_masks, axis=-1)
    pred_labels = np.argmax(predictions, axis=-1)

    if len(test_labels) != len(pred_labels):
        print(f"Inconsistent number of samples: true labels - {len(test_labels)}, predictions - {len(pred_labels)}")
        return test_images, test_masks, predictions

    conf_matrix = confusion_matrix(test_labels.flatten(), pred_labels.flatten())
    print("Confusion Matrix:\n", conf_matrix)

def visualize(img, mask, pred_image, location=None, date=None):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    img = np.array(img)
    if img.dtype != np.uint8:
        img = np.round(img).astype(np.uint8)
    
    axs[0, 0].imshow(cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(mask, cmap='gray')
    axs[0, 1].set_title('Mask')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(pred_image, cmap='gray')
    axs[1, 0].set_title('Predicted Image')
    axs[1, 0].axis('off')

    axs[1, 1].axis('off')

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
    return fig

def visualize_predictions(dataset, model, location=None, date=None, num_examples=3, fileDir=None):
    samples = random.sample(list(dataset), num_examples)
    
    imgs, masks = zip(*samples)
    imgs_array = np.array(imgs)

    predictions = model.predict(imgs_array)

    for i, (img, mask) in enumerate(samples):
        median = np.average(predictions[i])
        predictions[i] = np.where(predictions[i]<median, 0, 1)
        fig = visualize(img, mask, predictions[i], location, date)
        if fileDir:
            filename = f'prediction_{i}.png'
            save_plot(fig, filename, fileDir)
            print(f"Prediction {i} saved in {fileDir}/{filename}")

def evaluate(model_date, model_name, test_dataset, num_examples=1):
    print(f"Evaluating model from date: {model_date} with {num_examples} examples") 

    path =  f"models/{model_name}/Previous/{model_date}/"
    results_path = os.path.join(path,"results")
    model_path = os.path.abspath(os.path.join(path, "Model_Data", f"{model_name}_{model_date}.keras"))
    print(f"Loading Model from Path: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except ValueError as e:
        print(f"File not found error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")

    history_path = os.path.abspath(os.path.join(path,"Model_Data", "history.json"))
    
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

    if history is not None:
        evaluate_model(model, history, test_dataset, results_path)
        # visualize_predictions(test_dataset, model, num_examples=num_examples, fileDir=results_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the model with given date and number of examples.")
    parser.add_argument('--name', type = str, required=True, help = "The name of the model that is being evaluated ex: unet")
    parser.add_argument('-md', type=str, required=True, help='The date of the model to evaluate (format: YYYY-MM-DD-HH-MM).')
    parser.add_argument('-n', type=int, required=True, help='The number of examples to use for evaluation.')

    args = parser.parse_args()

    batch_size = 20
    buffer_size = 1000

    test_tfrecord_files = select_test_files()
    test_dataset = create_test_dataset(test_tfrecord_files, batch_size, buffer_size)
    evaluate(model_date=args.md, model_name= args.name ,test_dataset=test_dataset, num_examples=args.n)
