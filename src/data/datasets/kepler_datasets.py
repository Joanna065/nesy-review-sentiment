import logging
import math
from abc import ABCMeta
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm

from src.data.datasets.baseline_datasets import (
    BaselineDataset,
    Polemo2BaselineDataset,
    GoEmotionsBaselineDataset,
    MultiemoBaselineDataset,
)
from src.settings import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AVAILABLE_DATA_SOURCES = [
    'amuseWSD',
    'plWSD',
]


class KeplerDataset(BaselineDataset, metaclass=ABCMeta):
    PATH: Path
    NAME: str
    NUM_LABELS: int
    LANGUAGE: str
    DATA_SOURCE: str = 'amuseWSD'
    LABEL_COL: str = 'label'
    TEXT_COL: str = 'sentence'

    def __init__(
        self,
        split: str,
        processor: Callable,
    ):
        super().__init__(
            split=split,
            processor=processor,
        )
        self._load(split)

        assert len(self._texts) == len(self._labels)
        assert len(self._ids) == len(self._labels)

    def _load(self, split: str) -> None:
        data = self._load_json_data(split=split)

        self.total_plwn_ids = set()
        self._sample_plwn_ids = []
        ids = []
        for sample_id in tqdm(
            data.keys(),
            desc="Getting plWN ids set in dataset ...",
            total=len(data.keys()),
        ):
            ids.append(int(sample_id))
            sample_data = data[sample_id]
            tokens = sample_data['tokens']

            sample_plwn_ids_set = set()
            for token in tokens:
                if 'plwnSynsetId' in token and token['plwnSynsetId'] != "O":
                    plwn_id = int(token['plwnSynsetId'])
                    sample_plwn_ids_set.add(plwn_id)
                    self.total_plwn_ids.add(plwn_id)

            self._sample_plwn_ids.append(sample_plwn_ids_set)

        # after WSD a few samples are missing in some datasets (KLEJ_AR)
        self._ids = sorted(ids)
        self._labels = np.array(self._labels)[self._ids].tolist()
        self._texts = np.array(self._texts)[self._ids].tolist()

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
            self._sample_plwn_ids = [
                plwn_ids_set
                for idx, plwn_ids_set in enumerate(self._sample_plwn_ids)
                if idx in self._ids
            ]

            self._texts = self._texts[:size]
            self._ids = self._ids[:size]
            self._labels = self._labels[:size]
            self._sample_plwn_ids = self._sample_plwn_ids[:size]

            assert len(self._texts) == size
            assert len(self._ids) == size
            assert len(self._labels) == size
            assert len(self._sample_plwn_ids) == size

        else:
            (
                self._ids,
                _,
                self._texts,
                _,
                self._labels,
                _,
                self._sample_plwn_ids,
                _,
            ) = train_test_split(
                self._ids,
                self._texts,
                self._labels,
                self._sample_plwn_ids,
                train_size=size,
                shuffle=True,
                stratify=self._labels,
            )

        # get total plWN ids set in dataset
        self.total_plwn_ids = set().union(*self._sample_plwn_ids)


class IMDBKeplerDataset(KeplerDataset):
    PATH = DATA_DIR.joinpath('imdb')
    NAME = 'imdb'
    NUM_LABELS = 2
    LANGUAGE = 'en'


class MovieReviewsKeplerDataset(KeplerDataset):
    PATH = DATA_DIR.joinpath('movie_reviews')
    NAME = 'movie_reviews'
    NUM_LABELS = 2
    LANGUAGE = 'en'


class StanfordTreebankKeplerDataset(KeplerDataset):
    PATH = DATA_DIR.joinpath('stanford_treebank')
    NAME = 'stanford_treebank'
    NUM_LABELS = 5
    LANGUAGE = 'en'


class AllegroReviewsKeplerDataset(KeplerDataset):
    PATH = DATA_DIR.joinpath('klej_ar')
    NAME = 'klej_ar'
    NUM_LABELS = 5
    LANGUAGE = 'pl'
    LABEL_COL: str = 'rating'
    TEXT_COL: str = 'text'


class Polemo2KeplerDataset(Polemo2BaselineDataset, KeplerDataset):
    PATH = DATA_DIR.joinpath('polemo2')
    NAME = 'polemo2'
    NUM_LABELS = 4
    LANGUAGE = 'pl'
    LABEL_COL: str = 'label'
    TEXT_COL: str = 'text'

    DOMAIN = 'all'
    MODE = 'text'

    LABELS_MAP = {
        'meta_minus_m': 0,
        'meta_zero': 1,
        'meta_amb': 2,
        'meta_plus_m': 3,
    }


class MultiemoKeplerDataset(MultiemoBaselineDataset, KeplerDataset):
    PATH = DATA_DIR.joinpath('multiemo')
    NAME = 'multiemo'
    NUM_LABELS = 4
    LANGUAGE = 'en'
    LABEL_COL: str = 'label'
    TEXT_COL: str = 'text'

    DOMAIN = 'all'
    MODE = 'text'

    LABELS_MAP = {
        'meta_minus_m': 0,
        'meta_zero': 1,
        'meta_amb': 2,
        'meta_plus_m': 3,
    }


class GoEmotionsKeplerDataset(GoEmotionsBaselineDataset, KeplerDataset):
    PATH = DATA_DIR.joinpath('goemotions')
    NAME = 'goemotions'
    NUM_LABELS = 27  # multi-label
    LANGUAGE = 'en'
    MULTILABEL: bool = True
    LABEL_COL: str = 'labels'  # multi-label
    TEXT_COL: str = 'text'
