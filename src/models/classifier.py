from abc import ABCMeta, abstractmethod
from time import time

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Optimizer
from torchmetrics import MetricCollection, F1Score, Recall, Precision, Accuracy


class Classifier(pl.LightningModule, metaclass=ABCMeta):
    NAME: str

    def __init__(
        self,
        num_labels: int,
        multilabel: bool = False,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.multilabel = multilabel

        metrics = MetricCollection(
            {
                'accuracy': Accuracy(average='micro'),
                'precision': Precision(average='micro', num_classes=num_labels),
                'recall': Recall(average='micro', num_classes=num_labels),
                'f1_micro': F1Score(average='micro', num_classes=num_labels),
                'f1_macro': F1Score(average='macro', num_classes=num_labels),
                'f1_per_class': F1Score(average='none', num_classes=num_labels),
            }
        )
        self._train_metrics = metrics.clone(prefix='train/')
        self._val_metrics = metrics.clone(prefix='val/')
        self._test_metrics = metrics.clone(prefix='test/')

    @abstractmethod
    def step(
        self,
        batch: dict[str, Tensor],
        step_type: str,
    ) -> dict[str, Tensor]:
        ...

    def shared_step(
        self,
        batch: dict[str, Tensor],
        step_type: str,
    ) -> dict[str, Tensor]:
        output = self.step(batch, step_type=step_type)
        assert 'loss' in output
        assert 'logits' in output
        assert 'labels' in output
        return output

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        return self.shared_step(batch, step_type='train')

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        return self.shared_step(batch, step_type='val')

    def test_step(self, batch: dict[str, Tensor], batch_idx: int):
        return self.shared_step(batch, step_type='test')

    def on_train_epoch_start(self) -> None:
        self._epoch_start_time = time()

    def training_epoch_end(self, outputs) -> None:
        epoch_time = time() - self._epoch_start_time
        self.log('train/epoch_time', epoch_time, on_epoch=True, on_step=False)

        logits = torch.cat([out['logits'] for out in outputs]).float()
        labels = torch.cat([out['labels'] for out in outputs]).int()

        metrics = self._train_metrics(logits, labels)
        self._epoch_log_metrics(metrics, step_type=self._train_metrics.prefix)

    def validation_epoch_end(self, outputs) -> None:
        logits = torch.cat([out['logits'] for out in outputs]).float()
        labels = torch.cat([out['labels'] for out in outputs]).int()

        metrics = self._val_metrics(logits, labels)
        self._epoch_log_metrics(metrics, step_type=self._val_metrics.prefix)

    def test_epoch_end(self, outputs) -> None:
        logits = torch.cat([out['logits'] for out in outputs]).float()
        labels = torch.cat([out['labels'] for out in outputs]).int()

        metrics = self._test_metrics(logits, labels)
        self._epoch_log_metrics(metrics, step_type=self._test_metrics.prefix)

    def _epoch_log_metrics(self, metric_dict: dict[str, Tensor], step_type: str) -> None:
        f1_class_key = f'{step_type}f1_per_class'
        labels_str = [f'{f1_class_key}/{idx}' for idx in range(self.num_labels)]
        f1_class = metric_dict.pop(f1_class_key)
        metrics_per_class = dict(zip(labels_str, f1_class))

        metrics = metric_dict | metrics_per_class
        self.log_dict(metrics, on_epoch=True, on_step=False)

    @abstractmethod
    def configure_optimizers(self) -> Optimizer:
        ...
