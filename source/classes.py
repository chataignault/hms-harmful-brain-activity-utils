from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, List
import pandas as pd
import os
import numpy as np
import esig

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
        cols: List[str],
        depth: int,
        index: List[int],
        param_invariant: bool = False,
    ):
        """
        take the signature of selected columns,
        with max depth,
        and select only desired elements of the signature
        """
        if param_invariant:
            self.features.append(
                esig.stream2sig(
                    self.raw[cols].reset_index(drop=True).reset_index(), depth=depth
                )[index]
            )
        else:  # numerically instable
            self.features.append(esig.stream2sig(self.raw[cols], depth=depth)[index])
        self.feature_names += [f"sig-{idx}" for idx in index]
        return self


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
    def __init__(self, dir: Dir):
        self.eeg_chain = lambda sample: (
            EegChain()
            .open(Eeg(dir, sample))
            ._fillna()
            ._divide(coef=1000.0)
            .mean(cols=["Fp1", "EKG"])
            .var(cols=["F3", "EKG"])
            .signature(cols=["Fp1", "P3"], depth=3, index=range(6))
            .clip_all_(low=-1e4, high=1e4)
            .result()
        )
        self.spc_chain = None
        self.features = []

    def save(self, data: pd.DataFrame, path: Dir = Dir.intermediate_output):
        data.to_parquet(path)

    def process(self, metadata: pd.DataFrame, save: Optional[str] = None) -> np.ndarray:
        self.features.append(metadata.apply(self.eeg_chain, axis=1))
        X = pd.concat(self.features)
        if save:
            self.save(X, save)
        return X
