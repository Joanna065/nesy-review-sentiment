import logging
from typing import Optional, Type

import torch
from torch import Tensor
from pytorch_transformers import RobertaTokenizer

from src.data.datasets.senti_lare_datasets import SentiLAREDataset, SentiLAREInputFeatures
from src.data.main_datamodule import MainDataModule
from src.settings import SENTI_LARE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentiLAREDataModule(MainDataModule):
    CHECKPOINT_PATH = SENTI_LARE_DIR.joinpath('sentiLARE_pretrained')
    dataset_cls: Type[SentiLAREDataset]

    def __init__(
        self,
        dataset_cls: Type[SentiLAREDataset],
        tokenizer_name: str,
        use_plwn_sentiment: bool,
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

        self.tokenizer = RobertaTokenizer.from_pretrained(str(self.CHECKPOINT_PATH))
        self.use_plwn_sentiment = use_plwn_sentiment

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, 'fit'):
            self._train_ds = self.dataset_cls(
                split='train',
                processor=self.tokenizer,
                max_token_len=self.max_token_len,
                use_plwn_sentiment=self.use_plwn_sentiment,
            )
            if self.train_size:
                self._train_ds.set_length(size=self.train_size)

            self._val_ds = self.dataset_cls(
                split='dev',
                processor=self.tokenizer,
                max_token_len=self.max_token_len,
                use_plwn_sentiment=self.use_plwn_sentiment,
            )

        if stage in (None, 'test'):
            self._test_ds = self.dataset_cls(
                split='test',
                processor=self.tokenizer,
                max_token_len=self.max_token_len,
                use_plwn_sentiment=self.use_plwn_sentiment,
            )

    def collate(
        self,
        features: list[SentiLAREInputFeatures],
    ) -> dict[str, Tensor]:
        return {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.input_mask for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long),
            'pos_tag_ids': torch.tensor([f.pos_tag_ids for f in features], dtype=torch.long),
            'senti_word_ids': torch.tensor([f.sentiment_ids for f in features], dtype=torch.long),
            'polarity_ids': torch.tensor([f.polarity_ids for f in features], dtype=torch.long),
            'ids': torch.tensor([f.id for f in features], dtype=torch.long),
        }
