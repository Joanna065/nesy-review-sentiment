import logging
import math
from abc import ABCMeta
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm
from transformers import RobertaTokenizer

from src.data.datasets.baseline_datasets import (
    BaselineDataset,
    Polemo2BaselineDataset,
    GoEmotionsBaselineDataset,
    MultiemoBaselineDataset,
    GoEmotionsSentimentBaselineDataset,
)
from src.settings import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AVAILABLE_DATA_SOURCES = [
    'amuseWSD',
    'plWSD',
]
# verb(v), adjective(a), adverb(r), noun(n), others(u)
POS_TAG_IDS_MAP = {
    'VERB': 0,
    'ADJ': 1,
    'ADV': 2,
    'NOUN': 3,
    'OTHERS': 4,
}

SENTIMENT_MAP = {
    'negative': 0,
    'positive': 1,
    'neutral': 2,
}


@dataclass
class SentiLAREInput:
    words: list[str]
    pos_tag_ids: list[int]
    sentiment_ids: list[int]
    label: int
    id: int


@dataclass
class SentiLAREInputFeatures:
    input_ids: list[int]
    input_mask: list[int]
    segment_ids: list[int]
    pos_tag_ids: list[int]
    sentiment_ids: list[int]
    polarity_ids: list[int]
    label: int
    id: int


class SentiLAREDataset(BaselineDataset, metaclass=ABCMeta):
    PATH: Path
    NAME: str
    NUM_LABELS: int
    LANGUAGE: str
    DATA_SOURCE: str = 'amuseWSD'
    LABEL_COL: str = 'label'
    TEXT_COL: str = 'sentence'
    _processor: RobertaTokenizer

    def __init__(
        self,
        split: str,
        processor: RobertaTokenizer,
        max_token_len: int = 512,
        use_plwn_sentiment: bool = False,
    ):
        super().__init__(
            split=split,
            processor=processor,
        )
        self.use_plwn_sentiment = use_plwn_sentiment
        self._load(split)
        self.convert_examples_to_features_roberta(
            cls_token=self._processor.cls_token,
            sep_token=self._processor.sep_token,
            max_seq_length=max_token_len,
        )

        assert len(self._ids) == len(self._labels)
        assert len(self._ids) == len(self._sentilare_input_examples)

    def __getitem__(self, index: int):
        return self._features[index]

    def _load(self, split: str) -> None:
        data = self._load_json_data(split=split)

        self._sentilare_input_examples = []

        ids = []
        for sample_id in tqdm(
            data.keys(),
            desc="Processing dataset for sentiLARE...",
            total=len(data.keys()),
        ):
            id = int(sample_id)

            ids.append(id)
            sample_data = data[sample_id]
            tokens = sample_data['tokens']

            sample_words_list = []
            sample_sentiment_list = []
            sample_pos_list = []
            for token in tokens:
                word = token['text']
                sample_words_list.append(word)

                pos = token['pos']
                if pos in POS_TAG_IDS_MAP:
                    sample_pos_list.append(POS_TAG_IDS_MAP[pos])
                else:
                    sample_pos_list.append(4)  # others

                if self.use_plwn_sentiment and 'plwnSentimentScore' in token:
                    plwn_sentiment_score = float(token['plwnSentimentScore'])
                    sample_sentiment_list.append(plwn_sentiment_score)
                elif 'sentiwordnetScore' in token:
                    sentiwordnet_score = float(token['sentiwordnetScore'])
                    sample_sentiment_list.append(sentiwordnet_score)
                else:
                    sample_sentiment_list.append(0.0)

            assert len(sample_words_list) == len(sample_pos_list)
            assert len(sample_words_list) == len(sample_sentiment_list)

            sentilare_input_example = SentiLAREInput(
                id=int(sample_id),
                label=self._labels[id],
                words=sample_words_list,
                pos_tag_ids=sample_pos_list,
                # transform sentiment score (float) to integer
                sentiment_ids=[
                    1 if ele > 0 else 0 if ele < 0 else 2 for ele in sample_sentiment_list
                ],
            )
            self._sentilare_input_examples.append(sentilare_input_example)

        # after amuseWSD a few samples are missing in some datasets (KLEJ_AR)
        self._ids = sorted(ids)
        self._labels = np.array(self._labels)[self._ids].tolist()
        self._texts = None

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
            self._sentilare_input_examples = np.array(self._sentilare_input_examples)[
                self._ids
            ].tolist()

            self._ids = self._ids[:size]
            self._labels = self._labels[:size]
            self._sentilare_input_examples = self._sentilare_input_examples[:size]

            assert len(self._ids) == size
            assert len(self._labels) == size
            assert len(self._sentilare_input_examples) == size

        else:
            (self._ids, _, self._sentilare_input_examples, _, self._labels, _,) = train_test_split(
                self._ids,
                self._sentilare_input_examples,
                self._labels,
                train_size=size,
                shuffle=True,
                stratify=self._labels,
            )

    def convert_examples_to_features_roberta(
        self,
        cls_token: str,
        sep_token: str,
        max_seq_length: int = 512,
        sep_token_extra: bool = True,  # True for Roberta
        pad_token: int = 0,
        mask_padding_with_zero: bool = True,
    ) -> None:
        """
        Code adapted from
        https://github.com/thu-coai/SentiLARE/blob/5f1243788fb872e56b5e259939b932346b378419/finetune/sent_data_utils_sentilr.py

         Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        """

        self._features = []
        for example in self._sentilare_input_examples:
            words = example.words
            pos = example.pos_tag_ids
            sentiment = example.sentiment_ids
            label = example.label
            example_id = example.id

            tokens, poses, sentiments = [], [], []
            for i, word in enumerate(words):
                tok_list = self._processor.tokenize(word)
                tokens.extend(tok_list)
                poses.extend([pos[i]] * len(tok_list))
                sentiments.extend([sentiment[i]] * len(tok_list))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2

            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                poses = poses[: (max_seq_length - special_tokens_count)]
                sentiments = sentiments[: (max_seq_length - special_tokens_count)]

            # The convention in BERT is for single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as the "sentence vector". Note that this only makes sense because the entire model is fine-tuned.

            # 4 in POS tags means others, 2 in word-level polarity labels means neutral, and
            # 5 in sentence-level sentiment labels means unknown sentiment
            tokens = tokens + [sep_token]
            pos_ids = poses + [4]
            senti_ids = sentiments + [2]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                pos_ids += [4]
                senti_ids += [2]

            tokens = [cls_token] + tokens
            pos_ids = [4] + pos_ids
            senti_ids = [2] + senti_ids

            input_ids = self._processor.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            pos_ids = pos_ids + ([4] * padding_length)
            senti_ids = senti_ids + ([2] * padding_length)

            # During fine-tuning, the sentence-level label is set to unknown
            polarity_ids = [5] * max_seq_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(pos_ids) == max_seq_length
            assert len(senti_ids) == max_seq_length
            assert len(polarity_ids) == max_seq_length

            self._features.append(
                SentiLAREInputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=[0] * max_seq_length,
                    pos_tag_ids=pos_ids,
                    sentiment_ids=senti_ids,
                    polarity_ids=polarity_ids,
                    label=label,
                    id=example_id,
                )
            )


class IMDBSentiLAREDataset(SentiLAREDataset):
    PATH = DATA_DIR.joinpath('imdb')
    NAME = 'imdb'
    NUM_LABELS = 2
    LANGUAGE = 'en'


class MovieReviewsSentiLAREDataset(SentiLAREDataset):
    PATH = DATA_DIR.joinpath('movie_reviews')
    NAME = 'movie_reviews'
    NUM_LABELS = 2
    LANGUAGE = 'en'


class StanfordTreebankSentiLAREDataset(SentiLAREDataset):
    PATH = DATA_DIR.joinpath('stanford_treebank')
    NAME = 'stanford_treebank'
    NUM_LABELS = 5
    LANGUAGE = 'en'


class AllegroReviewsSentiLAREDataset(SentiLAREDataset):
    PATH = DATA_DIR.joinpath('klej_ar')
    NAME = 'klej_ar'
    NUM_LABELS = 5
    LANGUAGE = 'pl'
    LABEL_COL: str = 'rating'
    TEXT_COL: str = 'text'


class Polemo2SentiLAREDataset(Polemo2BaselineDataset, SentiLAREDataset):
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


class MultiemoSentiLAREDataset(MultiemoBaselineDataset, SentiLAREDataset):
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


class GoEmotionsSentiLAREDataset(GoEmotionsBaselineDataset, SentiLAREDataset):
    PATH = DATA_DIR.joinpath('goemotions')
    NAME = 'goemotions'
    NUM_LABELS = 27
    LANGUAGE = 'en'
    MULTILABEL: bool = True
    LABEL_COL: str = 'labels'  # multi-label
    TEXT_COL: str = 'text'


class GoEmotionsSentimentSentiLAREDataset(GoEmotionsSentimentBaselineDataset, SentiLAREDataset):
    PATH = DATA_DIR.joinpath('goemotions')
    NAME = 'goemotions_sent'
    NUM_LABELS = 4
    LANGUAGE = 'en'
    MULTILABEL: bool = True
    LABEL_COL: str = 'sent_labels'  # multi-label
    TEXT_COL: str = 'text'
