import pandas as pd
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

BEST_PARAMS = {
    'criterion': 'entropy', 
    'max_depth': 20, 'max_features': 
    'log2', 'min_samples_leaf': 4, 
    'min_samples_split': 5, 
    'n_estimators': 50
}
NUM_TRIALS = 1
OUTPUT_DIRECTORY = "results/best_1/"

def get_trained_model(X_train,y_train):
    #make model
    rf = RandomForestClassifier(verbose=1, n_jobs=16, **BEST_PARAMS)

    # train model
    rf.fit(X_train, y_train)

    return rf

if __name__ =="__main__":
    # make output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    # read in data with pickle
    X,y = pickle.load(open("../X_y.pkl", "rb"))

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    # intialize dataframe to hold metrics
    metrics = pd.DataFrame(columns=["Accuracy", "F1", "Precision", "Recall"])


    for i in range(NUM_TRIALS):
        # get the trained model
        rf = get_trained_model(X_train, y_train)

        # save model
        with open(os.path.join(OUTPUT_DIRECTORY, f"model_{i}.pkl"), 'wb') as f:
            pickle.dump(rf, f)

        # make predictions
        y_pred = rf.predict(X_test)

        # get metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        

        #add to dataframe
        metrics = metrics._append(
            {
                "Accuracy": accuracy,
                "F1": f1,
                "Precision": precision,
                "Recall": recall
            }, ignore_index=True
        )

    # write df to csv
    metrics.to_csv(os.path.join(OUTPUT_DIRECTORY, "metrics.csv"))

    # print out our average metrics
    print(metrics.mean())

    # print out our best metrics(based on f1)
    print(metrics.loc[metrics.F1.idxmax()])