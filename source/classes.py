from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import os

from .preamble import Const


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


class Eeg(Sample):
    def __init__(self, folder, sample: pd.Series):  # don't forget to add length in sec to the meta df
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
