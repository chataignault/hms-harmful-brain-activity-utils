import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from sklearn.utils import shuffle
from scipy.special import logit
from joblib import Parallel, delayed

from .preamble import Grade, Dir, KAGGLE, RANDOM_STATE, VOTE_COLS
from .classes import Eeg, FeatureGenerator


""" NOTES:
Extend signals with isna column so that no bad interpolation is done and information is not lost
EEG and spectrogram data : redundant ?

"""


def parquet_to_npy(in_folder: Dir, out_folder: Dir, eeg_id: str) -> None:
    eeg = pd.read_parquet(os.path.join(in_folder, f"{eeg_id}.parquet"))
    eeg = eeg.fillna(0.0)  # TODO
    eeg = eeg.values.astype("float32")
    np.save(os.path.join(out_folder, f"{eeg_id}.npy"), eeg)


def convert_parquet_to_npy(in_folder: Dir, out_folder: Dir, names: List[str]) -> None:
    """
    Convert all parquet files
    keeping the same name
    embarassingly parallel
    """
    Parallel(n_jobs=3, backend="loky")(
        delayed(parquet_to_npy)(in_folder, out_folder, eeg_id) for eeg_id in names
    )


def open_train_metadata(read: bool = False, checkna: bool = False) -> pd.DataFrame:
    """
    open and process train.csv file
    """
    if not KAGGLE and (
        not os.path.exists(os.path.join(Dir.intermediate_output, "meta_train_extended.parquet"))
        or not read
    ):
        train = pd.read_csv(os.path.join(Dir.root, "train.csv"))
        train["n_votes"] = train[VOTE_COLS].sum(axis=1)
        for c in VOTE_COLS:
            train[c] = train[c] / train["n_votes"]
        # TODO : what is the true length of each subsample ??
        train["eeg_length"] = (
            50
            # train["eeg_label_offset_seconds"].diff().shift(-1).fillna(-1).astype(int)
        )
        if not KAGGLE and checkna:
            train["contains_na"] = train.apply(
                lambda sub: Eeg(Dir.eeg_train, sub).open_subs().isna().any().any(),
                axis=1,
            )
    else:
        train = pd.read_parquet(
            os.path.join(Dir.intermediate_output, "meta_train_extended.parquet")
        )
    return train


def print_summary_metadata(data: pd.DataFrame) -> None:
    """
    Show general info from the sample of metadata
    """
    print("=" * 50)
    print("Metadata summary :")
    print("Len : ", len(data))
    summary_count = (
        data.groupby("expert_consensus")[["eeg_id"]].count().rename(columns={"eeg_id": "n_sample"})
    )
    tot = summary_count["n_sample"].sum()
    summary_count["percent"] = (summary_count["n_sample"] / tot * 100).astype(int)
    display(summary_count)  # noqa: F821
    print("=" * 50)


def pre_process_meta(
    meta: pd.DataFrame,
    y_cols: str,
    grade: Optional[Grade] = None,
) -> pd.DataFrame:
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


def process_target(Y: pd.DataFrame, classification: bool) -> pd.DataFrame:
    """
    if classification, return the majority class index
    otherwise return the logodds to apply regression
    """
    if classification:
        Y = pd.DataFrame(Y.idxmax(axis=1))
        return Y
    eps = 1e-5
    return logit(np.clip(Y, eps, 1 - eps))


def process_data_from_meta(
    meta: pd.DataFrame,
    feature_generator: FeatureGenerator,
    y_cols: str,
    max_nsample: Optional[int] = None,
    grade: Optional[Grade] = None,
    classification: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process the train data from the metadata to (design matrix,  target matrix)
    """
    meta = pre_process_meta(meta, y_cols, grade)
    Y = process_target(meta[y_cols], classification).iloc[:max_nsample]
    if feature_generator.parallel:
        X = feature_generator.parallel_process(meta.iloc[:max_nsample])
    else:
        X = feature_generator.process(meta.iloc[:max_nsample])
    return X, Y, meta.iloc[:max_nsample]
