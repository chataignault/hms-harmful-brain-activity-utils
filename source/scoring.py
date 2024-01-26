import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance

from .preamble import VOTE_COLS

def MSE_(Y:pd.DataFrame, Y_hat:pd.DataFrame, **kwargs) -> float:
    """
    Mean Squared Error function
    """
    return np.norm(Y - Y_hat)

def compute_wasserstein(predicted_probas, target_probas):
    ws = 0.
    for i in range(len(VOTE_COLS)):
        ws += wasserstein_distance(predicted_probas[:, i], target_probas.values[:, i])
    return ws

def score(Y:pd.DataFrame, Y_hat:pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Return summary of scores for true and predicted values
    TODO : implement Weiserstein distance and KL divergence
    """
    score_functions = {
        "MSE" :MSE_
    }
    scores = pd.DataFrame({
        sfn: score_functions[sfn](Y, Y_hat, **kwargs) for sfn in score_functions.keys()
    })
    return scores
    