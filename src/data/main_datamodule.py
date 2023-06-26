from typing import Optional, Type

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from src.data.datasets.base import BaseDataset
from src.data.datasets.baseline_datasets import BaselineDataset
from src.data.samplers import get_sampler


class MainDataModule(LightningDataModule):
    COLLATE_MAPPING = {
        'id': 'ids',
        'label': 'labels',
        'hurtbert_encoding': 'hurtbert_encodings',
    }

    def __init__(
        self,
        dataset_cls: Type[BaselineDataset],
        tokenizer_name: str,
        sampler_name: Optional[str] = None,
        batch_size: int = 16,
        num_workers: int = 0,
        train_size: Optional[int] = None,
        max_token_len: int = 512,
        **dataset_kwargs,
    ) -> None:
        super().__init__()

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._sampler_name = sampler_name
        self.dataset_cls = dataset_cls
        self.max_token_len = max_token_len
        self.tokenizer_name = tokenizer_name

        self.train_size = train_size
        self.dataset_kwargs = dataset_kwargs

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.collate_fn = DataCollatorWithPadding(self.tokenizer)

    def setup(self, stage: Optional[str] = None) -> None:

        if stage in (None, 'fit'):
            self._train_ds = self.dataset_cls(
                split='train',
                processor=self.tokenize,
            )
            if self.train_size:
                self._train_ds.set_length(size=self.train_size)

            self._val_ds = self.dataset_cls(
                split='dev',
                processor=self.tokenize,
            )

        if stage in (None, 'test'):
            self._test_ds = self.dataset_cls(
                split='test',
                processor=self.tokenize,
            )

    def tokenize(self, text: str) -> dict[str, Tensor]:
        return self.tokenizer(
            text=text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding=False,  # no padding, doing later
            return_attention_mask=True,
        )

    def collate(
        self,
        features: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        collated_features = self.collate_fn(features)
        collated_features = self._collate_mapping(collated_features)
        return collated_features

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.create_dataloader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.create_dataloader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.create_dataloader(self._test_ds, shuffle=False)

    def create_dataloader(self, dataset: BaseDataset, shuffle: bool):
        kwargs: dict = {}
        if shuffle and self._sampler_name:
            kwargs['sampler'] = get_sampler(
                sampler_name=self._sampler_name,
                labels=dataset.labels,
                batch_size=self._batch_size,
            )
        else:
            kwargs['shuffle'] = shuffle

        return self._get_dataloader(dataset, **kwargs)

    def _get_dataloader(self, dataset: Dataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            pin_memory=True,
            collate_fn=self.collate,
            **kwargs,
        )

    def _collate_mapping(
        self,
        features: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        for old, new in self.COLLATE_MAPPING.items():
            if old in features:
                if new in features:
                    raise ValueError(f"Feature {new} already exists.")
                features[new] = features.pop(old)
        return features
