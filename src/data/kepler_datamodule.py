from typing import Optional, Type, Any

import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from src.data.datasets.baseline_datasets import BaselineDataset
from src.data.datasets.kepler_datasets import KeplerDataset
from src.data.datasets.kepler_ke_dataset import KEDataset
from src.data.main_datamodule import MainDataModule


class KeplerDataModule(MainDataModule):
    dataset_cls: Type[KeplerDataset]

    def __init__(
        self,
        dataset_cls: Type[BaselineDataset],
        tokenizer_name: str,
        batch_size: int = 16,
        num_workers: int = 0,
        sampler_name: Optional[str] = None,
        train_size: Optional[int] = None,
        max_token_len: int = 512,
        **dataset_kwargs,
    ) -> None:
        super().__init__(
            dataset_cls=dataset_cls,
            tokenizer_name=tokenizer_name,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler_name=sampler_name,
            train_size=train_size,
            dataset_kwargs=dataset_kwargs,
            max_token_len=max_token_len,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, 'fit'):
            self._train_ds = self.dataset_cls(
                split='train',
                processor=self.tokenize,
            )
            if self.train_size:
                self._train_ds.set_length(size=self.train_size)

            self.dataset_kepler = KEDataset(
                train_sense_ids_set=self._train_ds.total_plwn_ids,
                processor=self.tokenize,
            )
            self.num_relations = len(self.dataset_kepler.relation_set)

            self._val_ds = self.dataset_cls(
                split='dev',
                processor=self.tokenize,
            )

        if stage in (None, 'test'):
            self._test_ds = self.dataset_cls(
                split='test',
                processor=self.tokenize,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return dict(
            task=super().train_dataloader(),
            kepler=DataLoader(
                dataset=self.dataset_kepler,
                num_workers=self._num_workers,
                batch_size=self._batch_size,
                pin_memory=True,
                collate_fn=self.collate_ke_dataset,
            ),
        )

    def collate_ke_dataset(
        self,
        features: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return dict(
            positive_heads=self.collate([feature['positive_head'] for feature in features]),
            positive_relations=torch.tensor([feature['positive_relation'] for feature in features]),
            positive_tails=self.collate([feature['positive_tail'] for feature in features]),
            negative_heads=[self.collate(feature['negative_heads']) for feature in features],
            negative_relations=[
                torch.tensor(feature['negative_relations']) for feature in features
            ],
            negative_tails=[self.collate(feature['negative_tails']) for feature in features],
        )
