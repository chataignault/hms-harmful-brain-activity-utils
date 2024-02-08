from matplotlib import pyplot as plt
from seaborn import heatmap
import numpy as np

from .preamble import VOTE_COLS


def plot_coefs(model):
    fig, axs = plt.subplots(nrows=2, figsize=(6, 12))
    heatmap(model.coef_, center=0.0, cmap="vlag", yticklabels=VOTE_COLS, ax=axs[0])
    heatmap(model.coef_ == 0.0, cmap="Reds", yticklabels=VOTE_COLS, ax=axs[1])
    axs[0].set_title("Coefficient values")
    n, m = model.coef_.shape
    axs[1].set_title(
        f"Lasso unselected coefs ({n*m - np.sum(np.sum(model.coef_==0.))} out of {n*m} remaining)"
    )
    return fig, axs


def plot_distributions(predicted_probas: np.array, target_probas: np.array):
    alpha = 0.5
    bins = np.linspace(0, 1, 40)
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
    for i in range(6):
        row, col = i % 3, i // 3
        axs[row, col].hist(
            predicted_probas[:, i],
            bins=bins,
            alpha=alpha,
            density=True,
            label="predicted",
        )
        axs[row, col].hist(target_probas[:, i], bins=bins, alpha=alpha, density=True, label="True")
        axs[row, col].legend()
        axs[row, col].set_title(f"Probability densities for {VOTE_COLS[i]}")
        axs[row, col].grid()
