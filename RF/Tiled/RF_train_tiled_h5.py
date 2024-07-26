import h5py
import numpy as np
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


SOURCE = 'data/512_features/features.h5'
OUTPUT_DIR = 'RF/Tiled/results/'
TRAIN_BATCHES = 20
VAL_BATCHES = 5
TEST_BATCHES = 3

def load_data_in_batches(h5_file, dataset_name, num_batches):
    with h5py.File(h5_file, 'r') as hf:
        data = hf[dataset_name]
        batch_size = data.shape[0] // num_batches
        for i in range(num_batches):
            # check if last batch
            if i == num_batches - 1:
                yield data[i * batch_size:]
                break
            start = i * batch_size
            end = (i + 1) * batch_size
            yield data[start:end]

def train_incremental_rf(h5_file,num_train_batches, num_val_batches, num_test_batches):
    # Define the Random Forest model
    params = {
        'criterion': 'entropy', 
        'max_depth': 20, 
        'max_features': 'log2', 
        'min_samples_leaf': 4, 
        'min_samples_split': 5,
    }
    rf = RandomForestClassifier(n_estimators=10, warm_start=True, n_jobs=10, **params)

    # Load and train on batches of training data
    for batch in load_data_in_batches(h5_file, 'train', num_train_batches):
        X_train = batch[:, :-1]
        y_train = batch[:, -1]
        rf.fit(X_train, y_train)
        rf.n_estimators += 10  # Increase the number of trees for the next batch

    # Validation
    val_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    for batch in load_data_in_batches(h5_file, 'validation', num_val_batches):
        X_val = batch[:, :-1]
        y_val = batch[:, -1]
        val_pred = rf.predict(X_val)
        val_metrics['accuracy'].append(accuracy_score(y_val, val_pred))
        val_metrics['precision'].append(precision_score(y_val, val_pred))
        val_metrics['recall'].append(recall_score(y_val, val_pred))
        val_metrics['f1'].append(f1_score(y_val, val_pred))

    # Test
    test_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    for batch in load_data_in_batches(h5_file, 'test', num_test_batches):
        X_test = batch[:, :-1]
        y_test = batch[:, -1]
        test_pred = rf.predict(X_test)
        test_metrics['accuracy'].append(accuracy_score(y_test, test_pred))
        test_metrics['precision'].append(precision_score(y_test, test_pred))
        test_metrics['recall'].append(recall_score(y_test, test_pred))
        test_metrics['f1'].append(f1_score(y_test, test_pred))

    # Calculate average metrics
    avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
    avg_test_metrics = {k: np.mean(v) for k, v in test_metrics.items()}
    
    return avg_val_metrics, avg_test_metrics, rf


if __name__ == "__main__":
    #make outut directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Train and evaluate the model
    avg_val_metrics, avg_test_metrics, rf = train_incremental_rf(SOURCE, TRAIN_BATCHES, VAL_BATCHES, TEST_BATCHES)

    # save model
    with open(os.path.join(OUTPUT_DIR, 'model_0.pkl'), 'wb') as f:
        pickle.dump(rf, f)

    # save metrics
    with open(os.path.join(OUTPUT_DIR, 'metrics_0.pkl'), 'wb') as f:
        pickle.dump({'val': avg_val_metrics, 'test': avg_test_metrics}, f)

    # Print the results
    print('Validation Metrics:')
    print(f"Accuracy: {avg_val_metrics['accuracy']}")
    print(f"Precision: {avg_val_metrics['precision']}")
    print(f"Recall: {avg_val_metrics['recall']}")
    print(f"F1 Score: {avg_val_metrics['f1']}")

    print('Test Metrics:')
    print(f"Accuracy: {avg_test_metrics['accuracy']}")
    print(f"Precision: {avg_test_metrics['precision']}")
    print(f"Recall: {avg_test_metrics['recall']}")
    print(f"F1 Score: {avg_test_metrics['f1']}")
