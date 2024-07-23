import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import pickle

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  #noqa
from sklearn.model_selection import HalvingGridSearchCV

OUTPUT_DIRECTORY = "results"

#load in the model with pickle
rsh = pickle.load(open("results/HalvingGridSearch.pkl", "rb"))

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
plt.savefig(os.path.join(OUTPUT_DIRECTORY, "ScoreCandidateOverIteration_Grid.png"))
plt.show()