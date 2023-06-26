import logging
from typing import Optional, Type

import numpy as np
import torch
from torch import Tensor

from src.data.datasets.hurtbert_embedding_datasets import HurtBertEmbeddingDataset
from src.data.main_datamodule import MainDataModule
from src.settings import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HurtBertEmbeddingDataModule(MainDataModule):
    dataset_cls: Type[HurtBertEmbeddingDataset]

    def __init__(
        self,
        dataset_cls: Type[HurtBertEmbeddingDataset],
        tokenizer_name: str,
        use_sentiwordnet: bool,
        batch_size: int = 16,
        num_workers: int = 0,
        max_token_len: int = 512,
        sampler_name: Optional[str] = None,
        train_size: Optional[int] = None,
        **dataset_kwargs,
    ) -> None:
        super().__init__(
            dataset_cls=dataset_cls,
            tokenizer_name=tokenizer_name,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler_name=sampler_name,
            train_size=train_size,
            max_token_len=max_token_len,
            dataset_kwargs=dataset_kwargs,
        )
        self.use_sentiwordnet = use_sentiwordnet

    def setup(self, stage: Optional[str] = None) -> None:
        logger.info("Reading HurtBert embedding .vec file...")
        hurtbert_embeddings_dict = self._read_embeddings(use_sentiwordnet=self.use_sentiwordnet)

        hurtbert_matrix = np.array(list(hurtbert_embeddings_dict.values()))
        logger.info(f'HurtBert embedding matrix shape: {hurtbert_matrix.shape}')

        # VOCAB x EMBEDDING_DIM -> self.embedding_weights
        self.embedding_weights = torch.from_numpy(hurtbert_matrix).float()

        if stage in (None, 'fit'):
            self._train_ds = self.dataset_cls(
                split='train',
                processor=self.tokenize,
                hurtbert_embeddings_dict=hurtbert_embeddings_dict,
            )
            if self.train_size:
                self._train_ds.set_length(size=self.train_size)

            self._val_ds = self.dataset_cls(
                split='dev',
                processor=self.tokenize,
                hurtbert_embeddings_dict=hurtbert_embeddings_dict,
            )

        if stage in (None, 'test'):
            self._test_ds = self.dataset_cls(
                split='test',
                processor=self.tokenize,
                hurtbert_embeddings_dict=hurtbert_embeddings_dict,
            )

    def collate(
        self,
        features: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        max_size = max((len(feature['hurtbert_embeddings']) for feature in features), default=0)
        hurtbert_embeddings = []
        for feature in features:
            diff = max_size - len(feature['hurtbert_embeddings'])
            hurtbert_embeddings.append(feature.pop('hurtbert_embeddings') + [0] * diff)

        return {
            **super().collate(features),
            'hurtbert_embeddings': torch.tensor(hurtbert_embeddings),
        }

    def _read_embeddings(self, use_sentiwordnet: bool) -> dict[str, np.ndarray]:
        hurtbert_embeddings = dict()
        vec_filepath = DATA_DIR.joinpath(f'hurtbert_embedding_sentiwordnet={use_sentiwordnet}.vec')

        with vec_filepath.open(mode='r') as f:
            vec_lines = [line.rstrip('\n') for line in f.readlines()]
            for line in vec_lines:
                pwn_id, vec = line.split('\t')
                one_hot_features = np.asarray(vec.split(' '), dtype='float32')
                hurtbert_embeddings[pwn_id] = one_hot_features

        return hurtbert_embeddings
