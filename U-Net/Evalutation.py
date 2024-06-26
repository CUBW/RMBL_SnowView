import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import tensorflow as tf
import datetime
import json

# Import created Classes
from Processing import Process

def save_plot(plt, filename, model_name):
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
    date_str = datetime.now().strftime("%Y-%m-%d")
    directory = os.path.join("/model", model_name, date_str,"results")
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
    # Extract training history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

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

    # Evaluate on training dataset
    train_loss = model.evaluate(train_dataset.batch(8))[0]
    # Evaluate on validation dataset
    val_loss_eval = model.evaluate(val_dataset.batch(8))[0]
    # Evaluate on test dataset
    test_loss = model.evaluate(test_dataset.batch(8))[0]

    print(f"Train loss: {train_loss}")
    print(f"Validation loss: {val_loss_eval}")
    print(f"Test loss: {test_loss}")

    # Generate predictions and confusion matrix
    predictions = model.predict(test_dataset.batch(8))
    test_labels_list = list(test_dataset.map(lambda x, y: y))
    test_labels = np.concatenate([label_batch.numpy() for label_batch in test_labels_list], axis=0)

    # Assuming test_labels are in one-hot encoded format
    test_labels = np.argmax(test_labels, axis=1)
    pred_labels = np.argmax(predictions, axis=1)

    if len(test_labels) != len(pred_labels):
        print(f"Inconsistent number of samples: true labels - {len(test_labels)}, predictions - {len(pred_labels)}")
        return

    conf_matrix = confusion_matrix(test_labels, pred_labels)
    print("Confusion Matrix:\n", conf_matrix)


    
    
    
if __name__ == "__main__":
    from Train import split_data
    
    
    # load the saved u-net model in /model/u-net_model.keras
    model = tf.keras.models.load_model("model/u-net_model.keras")
    model_name = "u-net"
    # load the processed dataset
    dataset = Process()
    train_dataset, val_dataset, test_dataset = split_data(dataset)
    evaluate_model(model, load_history(), train_dataset, val_dataset, test_dataset, model_name)
    