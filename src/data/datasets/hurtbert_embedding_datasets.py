import logging
import math
from abc import ABCMeta
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from torch import Tensor
from tqdm import tqdm

from src.data.datasets.baseline_datasets import (
    BaselineDataset,
    GoEmotionsBaselineDataset,
    Polemo2BaselineDataset,
    MultiemoBaselineDataset,
)
from src.settings import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AVAILABLE_DATA_SOURCES = [
    'amuseWSD',
    'plWSD',
]


class HurtBertEmbeddingDataset(BaselineDataset, metaclass=ABCMeta):
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
        hurtbert_embeddings_dict: dict[str, np.ndarray],
    ):
        super().__init__(
            split=split,
            processor=processor,
        )
        self._load(split, hurtbert_embeddings_dict=hurtbert_embeddings_dict)

        assert len(self._texts) == len(self._labels)
        assert len(self._ids) == len(self._labels)

    @property
    def labels(self) -> np.ndarray:
        return np.array(self._labels)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return dict(
            **self._processor(self._texts[index]),
            label=self._labels[index],
            id=self._ids[index],
            hurtbert_embeddings=self._hurtbert_embeddings[index],
        )

    def _load(
        self,
        split: str,
        hurtbert_embeddings_dict: dict[str, np.ndarray],
    ) -> None:
        # read annotated processed by WSD data -----------------------------------------
        data = self._load_json_data(split=split)

        # make hurtbert embedding per each word in text
        self._hurtbert_embeddings = []
        ids = []
        for sample_id in tqdm(
            data.keys(),
            desc="Preparing HurtBert embeddings...",
            total=len(data.keys()),
        ):
            ids.append(int(sample_id))
            sample_data = data[sample_id]
            tokens = sample_data['tokens']

            hurtbert_embedding_indices = []
            hurtbert_embeddings_keys = list(hurtbert_embeddings_dict.keys())
            for token in tokens:
                pwn_id = token['wnSynsetOffset']
                if pwn_id in hurtbert_embeddings_dict:
                    hurtbert_embedding_indices.append(hurtbert_embeddings_keys.index(pwn_id))
                else:
                    hurtbert_embedding_indices.append(0)

            self._hurtbert_embeddings.append(hurtbert_embedding_indices)

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
            self._hurtbert_embeddings = np.array(self._hurtbert_embeddings)[self._ids].tolist()

            self._texts = self._texts[:size]
            self._ids = self._ids[:size]
            self._labels = self._labels[:size]
            self._hurtbert_embeddings = self._hurtbert_embeddings[:size]
        else:
            (
                self._ids,
                _,
                self._texts,
                _,
                self._labels,
                _,
                self._hurtbert_embeddings,
                _,
            ) = train_test_split(
                self._ids,
                self._texts,
                self._labels,
                self._hurtbert_embeddings,
                train_size=size,
                shuffle=True,
                stratify=self._labels,
            )

        assert len(self._texts) == size
        assert len(self._ids) == size
        assert len(self._labels) == size
        assert len(self._hurtbert_embeddings) == size

    def __len__(self):
        return len(self._labels)


class IMDBHurtBertEmbeddingDataset(HurtBertEmbeddingDataset):
    PATH = DATA_DIR.joinpath('imdb')
    NAME = 'imdb'
    NUM_LABELS = 2
    LANGUAGE = 'en'


class MovieReviewsHurtBertEmbeddingDataset(HurtBertEmbeddingDataset):
    PATH = DATA_DIR.joinpath('movie_reviews')
    NAME = 'movie_reviews'
    NUM_LABELS = 2
    LANGUAGE = 'en'


class StanfordTreebankHurtBertEmbeddingDataset(HurtBertEmbeddingDataset):
    PATH = DATA_DIR.joinpath('stanford_treebank')
    NAME = 'stanford_treebank'
    NUM_LABELS = 5
    LANGUAGE = 'en'


class AllegroReviewsHurtBertEmbeddingDataset(HurtBertEmbeddingDataset):
    PATH = DATA_DIR.joinpath('klej_ar')
    NAME = 'klej_ar'
    NUM_LABELS = 5
    LANGUAGE = 'pl'
    LABEL_COL: str = 'rating'
    TEXT_COL: str = 'text'


class Polemo2HurtBertEmbeddingDataset(Polemo2BaselineDataset, HurtBertEmbeddingDataset):
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


class MultiemoHurtBertEmbeddingDataset(MultiemoBaselineDataset, HurtBertEmbeddingDataset):
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


class GoEmotionsHurtBertEmbeddingDataset(GoEmotionsBaselineDataset, HurtBertEmbeddingDataset):
    PATH = DATA_DIR.joinpath('goemotions')
    NAME = 'goemotions'
    NUM_LABELS = 27  # multi-label
    LANGUAGE = 'en'
    MULTILABEL: bool = True
    LABEL_COL: str = 'labels'  # multi-label
    TEXT_COL: str = 'text'
