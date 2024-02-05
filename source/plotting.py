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
