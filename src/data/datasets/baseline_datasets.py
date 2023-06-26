import json
import logging
import math
from abc import ABCMeta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from torch import Tensor
from unidecode import unidecode

from src.data.datasets.base import BaseDataset
from src.settings import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineDataset(BaseDataset, metaclass=ABCMeta):
    PATH: Path
    NAME: str
    NUM_LABELS: int
    LANGUAGE: str
    LABEL_COL: str = 'label'
    TEXT_COL: str = 'sentence'
    DATA_SOURCE: str = 'original'

    def __init__(
        self,
        split: str,
        processor: Callable,
    ) -> None:
        super().__init__()
        self._load_split_data(split)
        self._processor = processor

        assert len(self._texts) == len(self._labels)
        assert len(self._ids) == len(self._labels)

    @property
    def labels(self) -> np.ndarray:
        return np.array(self._labels)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return dict(
            **self._processor(self._texts[index]),
            label=self._labels[index],
            id=torch.tensor(self._ids[index]).long(),
        )

    def __len__(self):
        return len(self._labels)

    def _load_split_data(self, split: str) -> None:
        df = pd.read_csv(self.PATH.joinpath(f'{split}.tsv'), sep='\t', dtype={self.LABEL_COL: int})
        self._ids = df.index.values.tolist()
        self._texts = df[self.TEXT_COL].values.tolist()
        self._labels = df[self.LABEL_COL].values.tolist()

        labels_unique = sorted(df[self.LABEL_COL].unique())
        first_label = int(labels_unique[0])
        if first_label != 0:
            self._labels = [int(label - first_label) for label in self._labels]

    def _load_json_data(self, split: str) -> dict:
        if self.DATA_SOURCE == 'amuseWSD':
            json_path = self.PATH.joinpath(
                f'wsd_amuse_{self.LANGUAGE.upper()}', f'{split}.amuse_wsd.json'
            )
        else:
            json_path = self.PATH.joinpath(
                f'wsd_amuse_{self.LANGUAGE.upper()}', f'{split}.pl_wsd.json'
            )

        with json_path.open(mode='r') as f:
            data = json.load(f)
            logger.info("Read %d dataset records", len(data))

        return data

    def set_length(self, size: int):
        if self.MULTILABEL:
            test_size = len(self._labels) - size
            test_size_percentage = math.floor(test_size / len(self._labels))

            self._labels = np.array([np.array(xi) for xi in self._labels])
            self._ids = np.array(self._ids).reshape(-1, 1)

            self._ids, self._labels, _, _ = iterative_train_test_split(
                X=self._ids,
                y=self._labels,
                test_size=test_size_percentage,
            )
            self._ids = np.squeeze(self._ids).tolist()
            self._labels = self._labels.tolist()
            self._texts = np.array(self._texts)[self._ids].tolist()

            self._texts = self._texts[:size]
            self._ids = self._ids[:size]
            self._labels = self._labels[:size]

            assert len(self._texts) == size
            assert len(self._ids) == size
            assert len(self._labels) == size

        else:
            self._ids, _, self._texts, _, self._labels, _ = train_test_split(
                self._ids,
                self._texts,
                self._labels,
                train_size=size,
                shuffle=True,
                stratify=self._labels,
            )


class IMDBBaselineDataset(BaselineDataset):
    PATH = DATA_DIR.joinpath('imdb')
    NAME = 'imdb'
    NUM_LABELS = 2
    LANGUAGE = 'en'


class MovieReviewsBaselineDataset(BaselineDataset):
    PATH = DATA_DIR.joinpath('movie_reviews')
    NAME = 'movie_reviews'
    NUM_LABELS = 2
    LANGUAGE = 'en'


class StanfordTreebankBaselineDataset(BaselineDataset):
    PATH = DATA_DIR.joinpath('stanford_treebank')
    NAME = 'stanford_treebank'
    NUM_LABELS = 5
    LANGUAGE = 'en'


class AllegroReviewsBaselineDataset(BaselineDataset):
    PATH = DATA_DIR.joinpath('klej_ar')
    NAME = 'klej_ar'
    NUM_LABELS = 5
    LABEL_COL: str = 'rating'
    TEXT_COL: str = 'text'
    LANGUAGE = 'pl'


class Polemo2BaselineDataset(BaselineDataset):
    PATH = DATA_DIR.joinpath('polemo2')
    NAME = 'polemo2'
    NUM_LABELS = 4
    LABEL_COL: str = 'label'
    TEXT_COL: str = 'text'
    LANGUAGE = 'pl'

    DOMAIN = 'all'
    MODE = 'text'

    LABELS_MAP = {
        'meta_minus_m': 0,
        'meta_zero': 1,
        'meta_amb': 2,
        'meta_plus_m': 3,
    }

    def _load_split_data(self, split: str) -> None:
        path = self.PATH.joinpath('origin', f'{self.DOMAIN}.{self.MODE}.{split}.txt')
        with path.open() as f:
            lines = f.readlines()

        texts = []
        labels = []

        for line in lines:
            text, label = line.split('__label__')
            labels.append(label.rstrip('\n'))
            texts.append(text.strip())

        df = pd.DataFrame(list(zip(texts, labels)), columns=['text', 'label'])

        self._ids = df.index.values.tolist()
        self._texts = df[self.TEXT_COL].values.tolist()
        labels = df[self.LABEL_COL].values.tolist()
        self._labels = [self.LABELS_MAP[label] for label in labels]

    def _load_json_data(self, split: str) -> dict:
        if self.DATA_SOURCE == 'amuseWSD':
            json_path = self.PATH.joinpath(
                f'wsd_amuse_{self.LANGUAGE.upper()}',
                f'{self.DOMAIN}_{self.MODE}.{split}.amuse_wsd.json',
            )
        else:
            json_path = self.PATH.joinpath(
                f'wsd_amuse_{self.LANGUAGE.upper()}',
                f'{self.DOMAIN}_{self.MODE}.{split}.pl_wsd.json',
            )

        with json_path.open(mode='r') as f:
            data = json.load(f)
            logger.info("Read %d dataset records", len(data))

        return data


class MultiemoBaselineDataset(BaselineDataset):
    PATH = DATA_DIR.joinpath('multiemo')
    NAME = 'multiemo'
    NUM_LABELS = 4
    LABEL_COL: str = 'label'
    TEXT_COL: str = 'text'
    LANGUAGE = 'en'

    DOMAIN = 'all'
    MODE = 'text'

    LABELS_MAP = {
        'meta_minus_m': 0,
        'meta_zero': 1,
        'meta_amb': 2,
        'meta_plus_m': 3,
    }

    def _load_split_data(self, split: str) -> None:
        path = self.PATH.joinpath(f'{self.DOMAIN}.{self.MODE}.{split}.txt')
        with path.open(mode='r') as f:
            lines = f.readlines()

        texts = []
        labels = []

        for line in lines:
            text, label = line.split('__label__')
            labels.append(label.rstrip('\n'))
            texts.append(unidecode(text.strip()))

        df = pd.DataFrame(list(zip(texts, labels)), columns=['text', 'label'])

        self._ids = df.index.values.tolist()
        self._texts = df[self.TEXT_COL].values.tolist()
        labels = df[self.LABEL_COL].values.tolist()
        self._labels = [self.LABELS_MAP[label] for label in labels]


class GoEmotionsBaselineDataset(BaselineDataset):
    PATH = DATA_DIR.joinpath('goemotions')
    NAME = 'goemotions'
    NUM_LABELS = 27  # multi-label
    LANGUAGE = 'en'
    MULTILABEL: bool = True
    LABEL_COL: str = 'labels'  # multi-label
    TEXT_COL: str = 'text'

    def _load_split_data(self, split: str) -> None:
        df = pd.read_csv(self.PATH.joinpath(f'{split}.tsv'), sep='\t')
        self._ids = df.index.values.tolist()
        self._texts = df[self.TEXT_COL].values.tolist()
        multi_labels = df[self.LABEL_COL].values.tolist()
        multi_labels = [labels.split(',') for labels in multi_labels]

        one_hot_labels = []
        for multi_label in multi_labels:
            one_hot = [0.0] * self.NUM_LABELS
            for label in multi_label:
                one_hot[int(label) - 1] = 1.0

            one_hot_labels.append(one_hot)

        self._labels = one_hot_labels


class GoEmotionsSentimentBaselineDataset(BaselineDataset):
    PATH = DATA_DIR.joinpath('goemotions')
    NAME = 'goemotions_sent'
    NUM_LABELS = 4
    LANGUAGE = 'en'
    MULTILABEL: bool = True
    LABEL_COL: str = 'sent_labels'  # multi-label
    TEXT_COL: str = 'text'

    def _load_split_data(self, split: str) -> None:
        df = pd.read_csv(self.PATH.joinpath(f'{split}.tsv'), sep='\t')
        self._ids = df.index.values.tolist()
        self._texts = df[self.TEXT_COL].values.tolist()
        multi_labels = df[self.LABEL_COL].values.tolist()
        multi_labels = [labels.split(',') for labels in multi_labels]

        one_hot_labels = []
        for multi_label in multi_labels:
            one_hot = [0.0] * self.NUM_LABELS
            for label in multi_label:
                one_hot[int(label) - 1] = 1.0

            one_hot_labels.append(one_hot)

        self._labels = one_hot_labels
