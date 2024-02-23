import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import kl_div

from .preamble import VOTE_COLS


def MSE_(Y: np.ndarray, Y_hat: np.ndarray, **kwargs) -> float:
    """
    Mean Squared Error function
    """
    n, m = Y.shape
    N = n * m
    return np.linalg.norm(Y - Y_hat) / N


def compute_wasserstein(predicted_probas: np.ndarray, target_probas: np.ndarray) -> float:
    """
    Should be done on the joint probability actually
    """
    ws = 0.0
    for i in range(len(VOTE_COLS)):
        ws += wasserstein_distance(predicted_probas[:, i], target_probas[:, i])
    return ws


def compute_KL_div(predicted_probas: np.ndarray, target_probas: np.ndarray) -> float:
    """
    ISSUE with infinite values -> clip to zero for now
    (from test set, inf values are a small minority)
    """
    N = len(predicted_probas)
    kl_pointwise = kl_div(predicted_probas, target_probas)
    kl_pointwise[kl_pointwise == np.inf] = 0.0
    return np.sum(np.sum(kl_pointwise)) / N


def accuracy(predicted_probas: np.ndarray, target_probas: np.ndarray) -> float:
    true_label = np.argmax(target_probas, axis=1)
    pred_label = np.argmax(predicted_probas, axis=1)
    return np.sum(true_label == pred_label) / len(true_label)


def score(Y: np.ndarray, Y_hat: np.ndarray, **kwargs) -> pd.DataFrame:
    """
    Return summary of scores for true and predicted values
    - add cross-entropy
    """
    score_functions = {
        "MSE": MSE_,
        "Wasserstein": compute_wasserstein,
        "KullbackLDiv": compute_KL_div,
        "Accuracy": accuracy,
    }
    scores = {sfn: [score_functions[sfn](Y_hat, Y, **kwargs)] for sfn in score_functions.keys()}
    return pd.DataFrame(scores)
