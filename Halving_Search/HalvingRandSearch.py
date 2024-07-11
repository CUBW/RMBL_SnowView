import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import pickle

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  #noqa
from sklearn.model_selection import HalvingRandomSearchCV

OUTPUT_DIRECTORY = "results/"
SOURCE_FILEPATH = "../X_y.pkl"

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

def perfom_halving_rand_search(X, y, clf, save_results=True):
    # define distributions
    param_distributions = {
        "n_estimators": [10,20,30,40,50],
        "max_depth": [None, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"],
        "criterion": ["gini", "entropy"],
    }

    # make rand search
    rsh = HalvingRandomSearchCV(
        estimator=clf, 
        param_distributions=param_distributions, 
        resource='n_samples', 
        scoring='f1', 
        verbose = 1, 
        random_state=42,
        min_resources=3000
    )

    # fit rand search
    rsh.fit(X, y)

    # save the rand search
    if save_results:
        with open(OUTPUT_DIRECTORY + "HalvingRandSearch.pkl", 'wb') as f:
            pickle.dump(rsh, f)

    # return the rand search
    return rsh

def perform_analysis(rsh):
    # get the best params
    best_params = rsh.best_params_

    # get the best score
    best_score = rsh.best_score_

    # get the best index
    best_index = rsh.best_index_

    #write to file
    with open(OUTPUT_DIRECTORY + "HalvingRandSearchResults.txt", 'w') as f:
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best Score: {best_score}\n")
        f.write(f"Best Index: {best_index}\n")

    # get the cv results
    cv_results = rsh.cv_results_
    results = pd.DataFrame(cv_results)

    # graph 
    results["params_str"] = results.params.apply(str)
    results.drop_duplicates(subset=("params_str", "iter"), inplace=True)
    mean_scores = results.pivot(
        index="iter", columns="params_str", values="mean_test_score"
    )
    ax = mean_scores.plot(legend=False, alpha=0.6)

    labels = [
        f"iter={i}\nn_samples={rsh.n_resources_[i]}\nn_candidates={rsh.n_candidates_[i]}"
        for i in range(rsh.n_iterations_)
    ]

    ax.set_xticks(range(rsh.n_iterations_))
    ax.set_xticklabels(labels, rotation=45, multialignment="left")
    ax.set_title("Scores of candidates over iterations")
    ax.set_ylabel("mean test score", fontsize=15)
    ax.set_xlabel("iterations", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, "ScoreCandidateOverIteration.png"))

if __name__ == "__main__":
    print("Loading data...")
    # Load in data
    with open(SOURCE_FILEPATH, 'rb') as f:
        X, y = pickle.load(f)
    print("Data loaded.")

    # make output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    # make classifier
    clf = RandomForestClassifier(random_state=42)

    print("Performing Halving Rand Search...")
    start_time = time.time()
    # perform halving rand search
    rsh = perfom_halving_rand_search(X, y, clf)
    print("Halving Rand Search performed.")
    print("Rand Search saved at 'results/HalvingRandSearch.pkl'")
    print(f"Time taken: {format_time(time.time()-start_time)}")

    # perform analysis
    print("Performing analysis...")
    perform_analysis(rsh)
    print("Analysis performed.")