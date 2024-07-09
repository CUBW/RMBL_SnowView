import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Hyperparameter lists
PARAM_GRID = {
    'n_estimators': [10, 20, 25, 30, 40],
    'max_depth': [None, 5, 10, 20], # num features is 22
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

def format_time(seconds):
    """
    Formats the given time in seconds into a string representation.

    Args:
        seconds (float): The time in seconds.

    Returns:
        str: The formatted time string.

    Examples:
        >>> format_time(0.5)
        '0.500 s'
        >>> format_time(30)
        '30.00 s'
        >>> format_time(120)
        '02:00.0'
        >>> format_time(3660)
        '01:01:00.0'
    """
    if seconds < 1:
        return f"{seconds:.3f} s"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds %= 60
        return f"{minutes:02}:{seconds:04.1f}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds %= 60
        return f"{hours:02}:{minutes:02}:{seconds:04.1f}"

def data_prep():
    """
    Load and return the preprocessed data.

    Returns:
        The preprocessed data loaded from 'train_test_data.pkl'.
    """
    # Open data with pickle
    with open('train_test_data.pkl', 'rb') as f:
        return pickle.load(f)

def evaluate_model(param_name, param_value, clf):
    """
    Evaluates the performance of a classifier model on a given dataset.

    Args:
        param_name (str): The name of the parameter being evaluated.
        param_value: The value of the parameter being evaluated.
        clf: The classifier model to be evaluated.

    Returns:
        dict: A dictionary containing the evaluation metrics and other information.
            - 'param_name': The name of the parameter being evaluated.
            - 'param_value': The value of the parameter being evaluated.
            - 'accuracy': The accuracy score of the model.
            - 'precision': The precision score of the model.
            - 'recall': The recall score of the model.
            - 'f1_score': The F1 score of the model.
            - 'train_time': The training time of the model.
    """
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    return {
        'param_name': param_name,
        'param_value': param_value,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_time': end_time - start_time
    }

if __name__ == '__main__':

    start_time = time.time()

    # Prepare the data
    print("Preparing data...")
    X_train, X_test, y_train, y_test = data_prep()
    print("Data preparation completed.")
    print("Time taken:", format_time(time.time() - start_time))
    print()

    print("Starting hyperparameter sensitivity analysis...")

    # Initialize a list to store results
    results = []

    # Loop through each hyperparameter
    for param_name, param_values in PARAM_GRID.items():
        param_results = {
            'param_values': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        for param_value in tqdm(param_values):
            clf_params = {k: v[0] if isinstance(v, list) else v for k, v in PARAM_GRID.items()}
            clf_params[param_name] = param_value
            clf_params["verbose"] = 10
            clf_params["n_jobs"] = 16
            clf = RandomForestClassifier(**clf_params)
            
            results.append(evaluate_model(param_name, param_value, clf))
            
            param_results['param_values'].append(param_value)
            param_results['accuracy'].append(results[-1]['accuracy'])
            param_results['precision'].append(results[-1]['precision'])
            param_results['recall'].append(results[-1]['recall'])
            param_results['f1_score'].append(results[-1]['f1_score'])
        
        # Plot results
        plt.figure(figsize=(12, 8))
        plt.plot(param_results['param_values'], param_results['accuracy'], label='Accuracy')
        plt.plot(param_results['param_values'], param_results['precision'], label='Precision')
        plt.plot(param_results['param_values'], param_results['recall'], label='Recall')
        plt.plot(param_results['param_values'], param_results['f1_score'], label='F1 Score')
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title(f'Sensitivity Analysis for {param_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/{param_name}_sensitivity_analysis.png')
        plt.close()

    # Save all results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/master_results.csv')

    print("Hyperparameter sensitivity analysis completed. Results saved in the 'results' directory.")
    print("Total time taken:", format_time(time.time() - start_time))