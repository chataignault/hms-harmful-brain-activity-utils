import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import os
from sklearn.utils import shuffle

from .preamble import Grade, Dir, KAGGLE, RANDOM_STATE, VOTE_COLS
from .classes import Eeg


def open_train_metadata(folder: str) -> pd.DataFrame:
    """
    open and process train.csv file
    """
    if not KAGGLE and (not os.path.exists(os.path.join(Dir.intermediate_output, "meta_train_extended.parquet"))):
        train = pd.read_csv(os.path.join(Dir.root, "train.csv"))
        train["n_votes"] = train[VOTE_COLS].sum(axis=1)
        for c in VOTE_COLS:
            train[c] = train[c] / train["n_votes"]
        train["eeg_length"] = train["eeg_label_offset_seconds"].diff().shift(-1).fillna(-1).astype(int)
        if not KAGGLE:
            train["contains_na"] = train.apply(lambda sub: Eeg(Dir.eeg_train, sub).open_subs().isna().any().any(), axis=1)
    else:
        train = pd.read_parquet(os.path.join(Dir.intermediate_output, "meta_train_extended.parquet"))
    return train


def print_summary_metadata(data: pd.DataFrame) -> None:
    """
    Show general info from the sample of metadata
    """
    print("=" * 50)
    print("Metadata summary :")
    print("Len : ", len(data))
    summary_count = data.groupby("expert_consensus")[["eeg_id"]].count().rename(columns={"eeg_id": "n_sample"})
    tot = summary_count["n_sample"].sum()
    summary_count["percent"] = (summary_count["n_sample"] / tot * 100).astype(int)
    display(summary_count)
    print("=" * 50)


def pre_process_meta(meta: pd.DataFrame, y_cols: str, grade: Optional[Grade] = None, test_mode:bool=False) -> pd.DataFrame:
    """
    - make sure metadata can be sampled randomly or linearly without fear of class imbalance
    - subselection on the "quality" of the target variable : how much are experts agreeing on the subsamle
    """
    if KAGGLE:
        meta = shuffle(meta)
    else:  # deterministic output in local
        meta = shuffle(meta, random_state=RANDOM_STATE)

    if grade:
        meta = meta.loc[(meta[y_cols] >= grade).any(axis=1)]

    return meta


def process_target(Y: pd.DataFrame) -> pd.DataFrame:
    Y = pd.DataFrame(Y.idxmax(axis=1))
    return Y


def extract_features_eeg(eeg: pd.DataFrame) -> pd.DataFrame:
    m1 = eeg.mean(axis=0).T
    m1.index = pd.MultiIndex.from_product(iterables=[["Mean"], m1.index.values])
    m2 = eeg.var(axis=0).T
    m2.index = pd.MultiIndex.from_product(iterables=[["Var"], m2.index.values])
    return pd.concat([m1, m2])


def process_extracted_features_to_design(X_: List[pd.Series]) -> pd.DataFrame:
    """
    Take list of identically indexed pd.Series,
    Returns dataframe with :
        - each feature(signal) as column
        - sample number - ie index in X_ - as index
    (design matrix without intercept)
    """

    X = pd.DataFrame(columns=X_[0].index)
    for i, sample_features in enumerate(X_):
        X.loc[f"sample {i}"] = sample_features
    return X


def process_data_from_meta(
    meta: pd.DataFrame, y_cols: str, max_nsample: Optional[int] = None, grade: Optional[Grade] = None, test_mode:bool=False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process the train data from the metadata to (design matrix,  target matrix)
    """
    meta = pre_process_meta(meta, y_cols, grade, test_mode=test_mode)
    Y_all = process_target(meta[y_cols])
    n = len(Y_all)
    if max_nsample:
        n = np.min([n, max_nsample])
    X_, Y_ = [], []
    for j in range(n):
        sample = meta.iloc[j]
        if sample["eeg_length"] > 0:
            if "contains_na" in sample.index:
                if not sample["contains_na"]:
                    X_.append(extract_features_eeg(Eeg(Dir.eeg_train, sample).open_subs()))
                    Y_.append(Y_all.iloc[j])
            else:
                eeg = Eeg(Dir.eeg_train, sample).open_subs()
                if not eeg.isna().any().any():  # TODO : deal with Nan values
                    X_.append(extract_features_eeg(eeg))
                    Y_.append(Y_all.iloc[j])
    print("Number of samples without missing values selected : ", len(Y_))
    X = process_extracted_features_to_design(X_)
    Y = pd.concat(Y_, axis=0)
    return X, Y
