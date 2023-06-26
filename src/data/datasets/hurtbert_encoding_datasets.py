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

SENTIMENT_SCORES = [
    -0.8,
    -0.4,
    0.0,
    0.4,
    0.8,
]
# 8 basic emotions
EMOTION_NAMES = [
    'radość',  # joy (Ekhman)
    'strach',  # fear (Ekhman)
    'zaskoczenie',  # surprise (Ekhman)
    'smutek',  # sadness (Ekhman)
    'wstręt',  # disgust (Ekhman)
    'złość',  # anger (Ekhman)
    'zaufanie',  # trust (Plutchik)
    'cieszenie się',  # anticipation (Plutchik)
]
# 12 fundamental human values postulated by Puzynina (Puzynina, 1992)
EMOTION_VALUATIONS = [
    'użyteczność',  # utility
    'dobro drugiego człowieka',  # another's good
    'prawda',  # truth
    'wiedza',  # knowledge
    'piękno',  # beauty
    'szczęście',  # happiness
    'nieużyteczność',  # futility
    'krzywda',  # harm
    'niewiedza',  # ignorance
    'błąd',  # error
    'brzydota',  # ugliness
    'nieszczęście',  # misfortune
]


class HurtBertEncodingDataset(BaselineDataset, metaclass=ABCMeta):
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
        use_sentiwordnet: bool,
    ):
        super().__init__(
            split=split,
            processor=processor,
        )
        self._load(split, use_sentiwordnet)

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
            hurtbert_encoding=self._hurtbert_encodings[index],
        )

    def _load(self, split: str, use_sentiwordnet: bool) -> None:
        data = self._load_json_data(split=split)

        self._hurtbert_encodings = []
        ids = []
        for sample_id in tqdm(
            data.keys(),
            desc="Preparing HurtBert encodings...",
            total=len(data.keys()),
        ):
            ids.append(int(sample_id))
            sample_data = data[sample_id]
            tokens = sample_data['tokens']

            # create dict with feature keys in specific order
            encode_dict = dict()
            for sent_val in SENTIMENT_SCORES:
                encode_dict[sent_val] = 0
            for emo_name in EMOTION_NAMES:
                encode_dict[emo_name] = 0
            for emo_val in EMOTION_VALUATIONS:
                encode_dict[emo_val] = 0

            for token in tokens:
                # add sentiment
                sent_score = None
                if 'plwnSentimentScore' in token:
                    sent_score = float(token['plwnSentimentScore'])
                    encode_dict[sent_score] += 1
                if use_sentiwordnet and sent_score is None and 'sentiwordnetScore' in token:
                    sent_score = float(token['sentiwordnetScore'])

                    # threshold to plWN sentiment values
                    plwn_thresholds = np.array(SENTIMENT_SCORES)
                    plwn_sent_idx = np.argmin(np.abs(plwn_thresholds - sent_score))
                    sent_score = SENTIMENT_SCORES[plwn_sent_idx]
                    encode_dict[sent_score] += 1

                # add emotion names & valuations
                if 'plwnEmotionNames' in token:
                    emo_names = token['plwnEmotionNames']
                    for emo_name in emo_names:
                        encode_dict[emo_name] += 1

                if 'plwnEmotionValuations' in token:
                    emo_valuations = token['plwnEmotionValuations']
                    for emo_val in emo_valuations:
                        encode_dict[emo_val] += 1

            # delete neutral sentiment
            encode_dict.pop(0.0)
            self._hurtbert_encodings.append(list(encode_dict.values()))

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
            self._hurtbert_encodings = np.array(self._hurtbert_encodings)[self._ids].tolist()

            self._texts = self._texts[:size]
            self._ids = self._ids[:size]
            self._labels = self._labels[:size]
            self._hurtbert_encodings = self._hurtbert_encodings[:size]
        else:
            (
                self._ids,
                _,
                self._texts,
                _,
                self._labels,
                _,
                self._hurtbert_encodings,
                _,
            ) = train_test_split(
                self._ids,
                self._texts,
                self._labels,
                self._hurtbert_encodings,
                train_size=size,
                shuffle=True,
                stratify=self._labels,
            )

        assert len(self._texts) == size
        assert len(self._ids) == size
        assert len(self._labels) == size
        assert len(self._hurtbert_encodings) == size


class IMDBHurtBertEncodingDataset(HurtBertEncodingDataset):
    PATH = DATA_DIR.joinpath('imdb')
    NAME = 'imdb'
    NUM_LABELS = 2
    LANGUAGE = 'en'


class MovieReviewsHurtBertEncodingDataset(HurtBertEncodingDataset):
    PATH = DATA_DIR.joinpath('movie_reviews')
    NAME = 'movie_reviews'
    NUM_LABELS = 2
    LANGUAGE = 'en'


class StanfordTreebankHurtBertEncodingDataset(HurtBertEncodingDataset):
    PATH = DATA_DIR.joinpath('stanford_treebank')
    NAME = 'stanford_treebank'
    NUM_LABELS = 5
    LANGUAGE = 'en'


class AllegroReviewsHurtBertEncodingDataset(HurtBertEncodingDataset):
    PATH = DATA_DIR.joinpath('klej_ar')
    NAME = 'klej_ar'
    NUM_LABELS = 5
    LANGUAGE = 'pl'
    LABEL_COL: str = 'rating'
    TEXT_COL: str = 'text'


class Polemo2HurtBertEncodingDataset(Polemo2BaselineDataset, HurtBertEncodingDataset):
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


class MultiemoHurtBertEncodingDataset(MultiemoBaselineDataset, HurtBertEncodingDataset):
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


class GoEmotionsHurtBertEncodingDataset(GoEmotionsBaselineDataset, HurtBertEncodingDataset):
    PATH = DATA_DIR.joinpath('goemotions')
    NAME = 'goemotions'
    NUM_LABELS = 27  # multi-label
    LANGUAGE = 'en'
    MULTILABEL: bool = True
    LABEL_COL: str = 'labels'  # multi-label
    TEXT_COL: str = 'text'
