import numpy as np
import pandas as pd
import pickle
import time
import csv

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

RESULTS_FILEPATH = 'feature_results/feature_importances.csv'

def find_feature_importances(X, y, n_iterations=30, n_trees=25):
    """
    Calculates the average feature importances for a Random Forest Classifier

    Parameters:
    - X (DataFrame): The input features.
    - y (Series): The target variable.
    - n_iterations (int): The number of iterations to train the model (default: 30).
    - n_trees (int): The number of trees in the Random Forest Classifier (default: 25).

    Returns:
    - feature_importance_df (DataFrame): A DataFrame containing the feature importances.
    """
    start_time = time.time()
    verbose = 10

    # Placeholder for feature importances
    feature_importances = np.zeros(X.shape[1])

    #accuracy and f1 score
    avg_accuracy = 0
    avg_f1 = 0  

    # open csv
    f =  open(RESULTS_FILEPATH, mode='w')
    writer = csv.writer(f)
    writer.writerow(['Iteration', 'Time', 'Accuracy', 'F1 Score'] + list(X.columns))

    # Train the model multiple times
    for i in tqdm(range(n_iterations), unit='model'):
        iter_start_time = time.time()


        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        
        # Initialize the model
        rf = RandomForestClassifier(n_estimators=n_trees, random_state=i, n_jobs=16, verbose=verbose)
        
        # Train the model
        rf.fit(X_train, y_train)

        # Test Model
        pred = rf.predict(X_test)
        
        # get accuracy and f1 score
        accuracy = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        # Add the feature importances
        feature_importances += rf.feature_importances_

        # Add the accuracies
        avg_accuracy += accuracy
        avg_f1 += f1

        #record time and feature importances to csv
        writer.writerow([i, time.time() - iter_start_time, accuracy, f1] + list(rf.feature_importances_))

        verbose = 1

    
    # Average
    feature_importances /= n_iterations
    avg_accuracy /= n_iterations
    avg_f1 /= n_iterations

    #write averages to csv
    writer.writerow(['Average', time.time() - start_time, avg_accuracy, avg_f1] + list(feature_importances))

    #close writer
    f.close()

    # Create a DataFrame to display the feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return feature_importance_df


if __name__ == '__main__':
    X,y = pickle.load(open('X_y.pkl', 'rb'))

    # Number of times to train the model
    n_iterations = 30

    # Number of trees in the forest
    n_trees = 20

    feature_importance_df = find_feature_importances(X, y, n_iterations, n_trees)

    print(feature_importance_df)  
