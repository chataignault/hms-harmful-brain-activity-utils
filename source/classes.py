from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, List, Callable
import pandas as pd
import os
import numpy as np
from esig import tosig as ts
from .preamble import Const, Dir


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
        return Const.fq_eeg * start_s, Const.fq_eeg * (start_s + duration_s)

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

    def open(self, extension: str = ".parquet"):
        return pd.read_parquet(os.path.join(self.folder, str(self.eeg_id) + extension))

    def open_subs(self):
        return self.get_subs(self.open())

    def get_subs(self, sample: pd.DataFrame):
        start, end = self.get_start_end_subsample()
        return sample.iloc[start:end]


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
    def open(self, sample: Sample):
        self.raw = sample.open_subs()
        self.cols = self.raw.columns
        self.features = []
        return self

    def result(self) -> pd.Series:
        return pd.Series(np.concatenate(self.features), index=self.feature_names)

    ### PREPROCESS

    def _divide(self, coef: float):
        self.raw = self.raw / coef
        return self

    def _fillna(self, value: float = 0.0):
        self.raw = self.raw.fillna(value)
        return self

    ### POSTPROCESS

    def clip_all_(self, low: float, high: float):
        self.features = [np.clip(f, low, high) for f in self.features]
        return self

    ### FEATURES

    def mean(self, cols: List[str]):
        if cols:
            self.features.append(np.mean(self.raw[cols], axis=0))
            self.feature_names += [col + "-mean" for col in cols]
        else:
            self.features.append(np.mean(self.raw, axis=0))
            self.feature_names += [col + "-mean" for col in self.cols]
        return self

    def var(self, cols: List[str]):
        if cols:
            self.features.append(np.var(self.raw[cols], axis=0))
            self.feature_names += [col + "-var" for col in cols]
        else:
            self.features.append(np.var(self.raw, axis=0))
            self.feature_names += [col + "-var" for col in self.cols]
        return self

    def signature(
        self,
        depth: int,
        cols: Optional[List[str]] = None,
        index: Optional[List[int]] = None,
        parameterise: bool = False,
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
        if parameterise:
            index = [
                i
                for i in index
                if i not in ChainBuilder.t_dependant_sig_indexes(len(cols), depth)
            ]
            self.features.append(
                ts.stream2sig(
                    self.raw[cols].reset_index(drop=True).reset_index().values, depth
                )[index]
            )

        else:  # numerically instable
            self.features.append(ts.stream2sig(self.raw[cols].values, depth)[index])
        # print(ts.sigkeys(len(cols), depth))
        sig_index = np.array(ts.sigkeys(len(cols), depth).strip().split(" "))[index]
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
    def __init__(self, eeg_chain: Callable):
        self.eeg_chain = eeg_chain
        self.spc_chain = None
        self.features = []

    def save(self, data: pd.DataFrame, path: Dir = Dir.intermediate_output):
        data.to_parquet(path)

    def process(self, metadata: pd.DataFrame, save: Optional[str] = None) -> np.ndarray:
        self.features.append(metadata.apply(self.eeg_chain, axis=1))
        X = pd.concat(self.features)
        X.index = metadata["eeg_id"]
        if save:
            self.save(X, save)
        return X
