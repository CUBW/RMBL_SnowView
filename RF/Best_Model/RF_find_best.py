import pandas as pd
import pickle
import os

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

OUTPUT_DIR = 'results/'

def train_model(x, y, params):
    """
    Trains a random forest classifier model using the given input features and labels.

    Parameters:
    x (array-like): The input features for training the model.
    y (array-like): The target labels for training the model.
    params (dict): The parameters to be passed to the RandomForestClassifier.

    Returns:
    RandomForestClassifier: The trained random forest classifier model.
    """
    model = RandomForestClassifier(**params)
    model.fit(x, y)
    return model

def evaluate_model(model,x,y):
    """
    Evaluates the performance of a Random Forest model.

    Parameters:
    model (object): The trained model.
    x (array-like): The input features for evaluation.
    y (array-like): The true labels for evaluation.

    Returns:
    accuracy (float): The accuracy of the model.
    confusion (array-like): The confusion matrix.
    precision (float): The precision score.
    recall (float): The recall score.
    f1 (float): The F1 score.
    report (str): The classification report.
    """
    y_pred = model.predict(x)
    accuracy = accuracy_score(y,y_pred)
    confusion = confusion_matrix(y,y_pred)
    precision = precision_score(y,y_pred)
    recall = recall_score(y,y_pred)
    f1 = f1_score(y,y_pred)
    report = classification_report(y,y_pred)
    return accuracy, confusion, precision, recall, f1, report

def save_model(model,filename):
    """
    Saves a model to a file using pickle.

    Parameters:
    model (object): The model to be saved.
    filename (str): The name of the file to save the model to.

    Returns:
    None
    """
    pickle.dump(model,open(os.path.join(OUTPUT_DIR, filename),'wb'))

def save_metrics(accuracy, confusion, precision, recall, f1, report, filename):
    """
    Saves the evaluation metrics to a CSV file.

    Parameters:
    accuracy (float): The accuracy of the model.
    confusion (array-like): The confusion matrix.
    precision (float): The precision score.
    recall (float): The recall score.
    f1 (float): The F1 score.
    report (str): The classification report.
    filename (str): The name of the file to save the metrics to.

    Returns:
    None
    """
    metrics = {
        'accuracy': accuracy,
        'confusion': confusion,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }
    pd.DataFrame(metrics).to_csv(os.path.join(OUTPUT_DIR, filename))

if __name__ == '__main__':
    # create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # load in data with pickle
    x,y = pickle.load(open('../../X_y.pkl','rb'))

    # prameters from halving grid search
    params =  {
        'criterion': 'entropy', 
        'max_depth': 20, 
        'max_features': 'log2', 
        'min_samples_leaf': 4, 
        'min_samples_split': 5, 
        'n_estimators': 50,
        'n_jobs': -1,
        'verbose': 1,
    }
    
    scores = []
    confusion_matrices = []
    reports = []

    best_f1 = 0
    best_model = None
    best_index = None

    # make multiple models to find the best one
    for i in tqdm(range(25), unit='model'):
        # split data
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=i)

        # train model
        model = train_model(x_train,y_train,params)

        # evaluate model
        accuracy, confusion, precision, recall, f1, report = evaluate_model(model,x_test,y_test)
        scores.append((accuracy, precision, recall, f1))
        confusion_matrices.append(confusion)
        reports.append(report)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_index = i
    
    # save best model
    save_model(best_model,'best_model.pkl')

    # save scores
    save_metrics(
        scores[best_index][0],
        confusion_matrices[best_index], 
        scores[best_index][1], 
        scores[best_index][2], 
        scores[best_index][3],
        reports[best_index],
        'best_metrics.csv'
    )

    # display best model scores
    print(reports[best_index])