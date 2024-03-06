from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, List, Callable
import pandas as pd
import os
import numpy as np
from math import factorial
from esig import tosig as ts
from joblib import Parallel, delayed
from .preamble import Const, EEG_COLS


class Sample(ABC):
    def __init__(self, folder: str):
        self.folder = folder

    @abstractmethod
    def open(self, **kwargs) -> pd.DataFrame:
        NotImplemented

    @abstractmethod
    def open_subs(self, subsample: int) -> pd.DataFrame:
        NotImplemented

    def get_start_end_subsample(self) -> Tuple[int, int]:
        start_s = int(self.eeg_label_offset_seconds)
        duration_s = int(self.eeg_length)
        return int(Const.fq_eeg) * start_s, int(Const.fq_eeg) * (start_s + duration_s)

    def plot(self, columns: Optional[Union[str, List[str]]] = None):
        kwargs = {
            "figsize": (10, 5),
            "grid": True,
            "legend": True,
            "title": f"sample {self.eeg_id}",
            "alpha": 0.7,
        }
        if columns:
            self.open()[columns].plot(**kwargs)
        else:
            self.open().plot(**kwargs)


class Eeg(Sample):
    def __init__(
        self, folder, sample: pd.Series
    ):  # don't forget to add length in sec to the meta df
        super().__init__(folder)
        kwargs = sample.to_dict()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def open(self):
        return pd.read_parquet(os.path.join(self.folder, str(int(self.eeg_id)) + ".parquet"))

    def open_npy(self):
        return np.load(os.path.join(self.folder, str(int(self.eeg_id)) + ".npy"))

    def open_subs(self):
        return self.get_subs(self.open())

    def open_subs_npy(self):
        return self.get_subs_np(self.open_npy())

    def get_subs(self, sample: pd.DataFrame) -> pd.DataFrame:
        start, end = self.get_start_end_subsample()
        return sample.iloc[start:end]

    def get_subs_np(self, sample: np.ndarray) -> np.ndarray:
        start, end = self.get_start_end_subsample()
        return sample[start:end]


class Spect(Sample):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

    def open():
        None

    def open_subsample(self, subsample: int):
        None


class ChainBuilder:
    """
    Base class to define a feature extraction
    Append each feature as a numpy array
    Track feature names
    """

    def __init__(self):
        self.raw = None
        self.features = None
        self.feature_names = []

    ### I/O
    def open(self, sample: Sample, subsample: bool = True):
        if subsample:
            self.raw = sample.open_subs()
        else:
            self.raw = sample.open()
        self.cols = self.raw.columns
        self.features = []
        return self

    def open_npy(self, sample: Sample, subsample: bool = True):
        if subsample:
            self.raw = sample.open_subs_npy()
        else:
            self.raw = sample.open_npy()
        self.cols = EEG_COLS
        self.features = []
        return self

    def result(self) -> pd.Series:
        return pd.Series(np.concatenate(self.features), index=self.feature_names)

    def result_npy(self) -> np.ndarray:
        return np.concatenate(self.features)

    ### PREPROCESS

    def _center(self):
        self.raw = self.raw - np.mean(self.raw, axis=0)
        return self

    # def _select_first_samples(self, n_samples:int):

    def _divide(self, coef: float):
        self.raw = self.raw / coef
        return self

    def _fillna(self, value: float = 0.0):
        self.raw = self.raw.fillna(value)
        return self

    # def _fillna_np(self, value: float = 0.0):
    #     self.raw = np.fillself.raw.fillna(value)
    #     return self

    ### POSTPROCESS

    def clip_all_(self, low: float, high: float):
        self.features = [np.clip(f, low, high) for f in self.features]
        return self

    ### FEATURES

    def mean(self, cols: List[str]):
        if cols:
            self.features.append(np.mean(self.raw[cols], axis=0))
            self.feature_names += [str(col) + "-mean" for col in cols]
        else:
            self.features.append(np.mean(self.raw, axis=0))
            self.feature_names += [str(col) + "-mean" for col in self.cols]
        return self

    def mean_npy(self, cols: List[str]):
        if cols:
            self.features.append(np.mean(self.raw[:, cols], axis=0))
            self.feature_names += [str(col) + "-mean" for col in cols]
        else:
            self.features.append(np.mean(self.raw, axis=0))
            self.feature_names += [str(col) + "-mean" for col in self.cols]
        return self

    def var(self, cols: List[str]):
        if cols:
            self.features.append(np.var(self.raw[cols], axis=0))
            self.feature_names += [str(col) + "-var" for col in cols]
        else:
            self.features.append(np.var(self.raw, axis=0))
            self.feature_names += [str(col) + "-var" for col in self.cols]
        return self

    def var_npy(self, cols: List[str]):
        if cols:
            self.features.append(np.var(self.raw[:, cols], axis=0))
            self.feature_names += [str(col) + "-var" for col in cols]
        else:
            self.features.append(np.var(self.raw, axis=0))
            self.feature_names += [str(col) + "-var" for col in self.cols]
        return self

    def signature(
        self,
        depth: int,
        cols: Optional[List[str]] = None,
        index: Optional[List[int]] = None,
        time_augment: bool = False,
    ):
        """
        take the signature of selected columns,
        with max depth,
        and select only desired elements of the signature
        """
        if cols is None:
            cols = self.raw.columns
        if index is None:
            p = len(cols)
            n_sig_terms = ChainBuilder.n_sig_coordinates(p, depth)
            index = range(n_sig_terms)
        if time_augment:
            index = [
                i for i in index if i not in ChainBuilder.t_dependant_sig_indexes(len(cols), depth)
            ]
            self.features.append(
                ts.stream2sig(
                    np.concatenate(
                        [
                            np.linspace(0.0, 50, len(self.raw)).reshape(-1, 1),
                            self.raw[cols],  # .values,
                        ],
                        axis=1,
                    ),
                    depth,
                )[index]
            )

        else:  # numerically instable
            self.features.append(ts.stream2sig(self.raw[cols].values, depth)[index])
        sig_index = np.array(ts.sigkeys(len(cols), depth).strip().split(" "))[index]
        self.feature_names += [f"sig-{idx}" for idx in sig_index]
        return self

    def signature_npy(
        self,
        depth: int,
        cols: List[int],
        index: Optional[List[int]] = None,
        time_augment: bool = False,
        factorial_rescale: bool = False,
    ):
        """
        take the signature of selected columns,
        with max depth,
        and select only desired elements of the signature
        if rescale, multiply by k! to avoid decay
        """
        p = len(cols) + int(time_augment)
        if index is None:
            n_sig_terms = ChainBuilder.n_sig_coordinates(p, depth)
            index = range(n_sig_terms)
        if time_augment:
            index = [i for i in index if i not in ChainBuilder.t_dependant_sig_indexes(p, depth)]
            sig = ts.stream2sig(
                np.concatenate(
                    [
                        np.linspace(0.0, Const.eeg_len, len(self.raw)).reshape(-1, 1),
                        self.raw[:, cols],
                    ],
                    axis=1,
                ),
                depth,
            )[index]
        else:  # numerically instable
            sig = ts.stream2sig(self.raw[:, cols], depth)[index]
        if factorial_rescale:
            sig = ChainBuilder.factorial_rescale(sig, depth, p)
        self.features.append(sig)
        sig_index = np.array(ts.sigkeys(p, depth).strip().split(" "))[index]
        self.feature_names += [f"sig-{idx}" for idx in sig_index]
        return self

    @staticmethod
    def n_sig_coordinates(p: int, depth: int) -> int:
        return int((p ** (depth + 1) - 1) / (p - 1))

    @staticmethod
    def t_dependant_sig_indexes(p: int, depth: int) -> set:
        """
        returns the set of indices of the signature that will be iterated only
        on the time variable, in case of forcing parameterisation dependance
        p : path dimension
        """
        return set(np.cumsum([(p + 1) ** k for k in range(depth)]))

    @staticmethod
    def factorial_rescale(sig: np.array, depth: int, p: int) -> np.ndarray:
        fact = np.array(
            [1.0] + [factorial(d) for d in range(1, depth + 1) for _ in range(p**d - 1)]
        )
        return np.multiply(sig, fact)  # r


class EegChain(ChainBuilder):
    """
    Add methods specific to EEGs
    """

    def __init__(self):
        super().__init__()

    def special_eeg_method(self):
        None
        return self


class SpcChain(ChainBuilder):
    """
    Add methods specific to spectrograms
    """

    def __init__(self, eeg: Eeg):
        super().__init__(eeg)

    def special_spc_method(self):
        None
        return self


class FeatureGenerator:
    def __init__(self, eeg_chain: Callable, parallel: bool = False, save: Optional[str] = None):
        self.eeg_chain = eeg_chain
        self.spc_chain = None
        self.features = []
        self.save = save
        self.parallel = parallel  # just a flag

    def _save(self, path: str):
        self.features.to_parquet(path)

    def _get_index_ids(self, meta: pd.DataFrame):
        """
        Get the right index from metadata
            - train data has eeg id and sub_id
            - test data has only id
        """
        if "eeg_sub_id" in meta.columns:
            return pd.MultiIndex.from_frame(meta[["eeg_id", "eeg_sub_id"]])
        return meta["eeg_id"]

    def process(self, metadata: pd.DataFrame, save: Optional[str] = None) -> pd.DataFrame:
        """
        Compute features iteratively on each subsample,
        keep track of (eeg_id, eeg_sub_id) which is a primary key
        """
        self.features = metadata.apply(self.eeg_chain, axis=1)
        self.features.index = self._get_index_ids(metadata)
        if save or self.save:
            path = self.save if save is None else save
            self._save(path=path)
        return self.features

    def parallel_process(self, metadata: pd.DataFrame, save: Optional[str] = None) -> np.ndarray:
        """
        Compute features faster with .npy extension and parallelisation
        """
        columns = ["eeg_id", "eeg_sub_id", "eeg_label_offset_seconds", "eeg_length"]
        self.features = np.array(
            Parallel(n_jobs=4, backend="loky")(
                delayed(self.eeg_chain)(eeg) for _, eeg in metadata[columns].iterrows()
            )
        )
        # self.features.index = self._get_index_ids(metadata)
        if save or self.save:
            path = self.save if save is None else save
            self._save(path=path)
        return self.features
