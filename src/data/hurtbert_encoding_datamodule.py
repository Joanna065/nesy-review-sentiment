import logging
from typing import Optional, Type

from src.data.datasets.hurtbert_encoding_datasets import HurtBertEncodingDataset
from src.data.main_datamodule import MainDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HurtBertEncodingDataModule(MainDataModule):
    dataset_cls: Type[HurtBertEncodingDataset]

    def __init__(
        self,
        dataset_cls: Type[HurtBertEncodingDataset],
        tokenizer_name: str,
        use_sentiwordnet: bool,
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

        self.use_sentiwordnet = use_sentiwordnet

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, 'fit'):
            self._train_ds = self.dataset_cls(
                split='train',
                processor=self.tokenize,
                use_sentiwordnet=self.use_sentiwordnet,
            )
            if self.train_size:
                self._train_ds.set_length(size=self.train_size)

            self._val_ds = self.dataset_cls(
                split='dev',
                processor=self.tokenize,
                use_sentiwordnet=self.use_sentiwordnet,
            )

        if stage in (None, 'test'):
            self._test_ds = self.dataset_cls(
                split='test',
                processor=self.tokenize,
                use_sentiwordnet=self.use_sentiwordnet,
            )
