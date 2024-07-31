import pandas as pd
import numpy as np
import pickle
import time

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

start_time = time.time()
#import df from file using pickle
images_df = pd.read_pickle('640_df.pkl')

#randomly select 50% of the data for use in the grid search
images_df = images_df.sample(frac=0.5)

#lets do a grid search
param_dist = {
    'n_estimators': [5, 10, 15, 25, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

#set up model
rf = RandomForestClassifier(random_state=42)

print("Starting rand search")

# Set up the grid search
rand_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, scoring='accuracy',
                           cv=5, n_jobs=8, verbose=3)

#get data
X = images_df.drop(labels=['labels'], axis=1)
y = images_df['labels'].values

# Fit the grid search to the data
rand_search.fit(X, y)

#save everything to disk when done fitting
pickle.dump(rand_search, open('rand_search.pkl', 'wb'))
print("Finished, took ", time.time()-start_time, " seconds")