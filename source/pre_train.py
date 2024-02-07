import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


def extract_validation_set(
    all: pd.DataFrame, ratio: float = 0.1, seed: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(all, test_size=ratio, shuffle=True, random_state=seed)
