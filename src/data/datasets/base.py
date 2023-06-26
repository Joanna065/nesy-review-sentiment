from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


class BaseDataset(Dataset, metaclass=ABCMeta):
    PATH: Path
    NAME: str
    NUM_LABELS: int
    LANGUAGE: str
    DATA_SOURCE: str
    MULTILABEL: bool = False

    @property
    @abstractmethod
    def labels(self) -> np.ndarray:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        ...

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def set_length(self, size: int) -> None:
        ...
