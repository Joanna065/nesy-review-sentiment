import time
from datetime import timedelta

from pytorch_lightning.callbacks import Callback


class TrainDurationCallback(Callback):
    def on_train_start(self, trainer, pl_module) -> None:
        self.start_time = time.monotonic()

    def on_train_end(self, trainer, pl_module, unused=None):
        end_time = time.monotonic()
        diff = timedelta(seconds=end_time - self.start_time)
        diff_seconds = diff.total_seconds()
        trainer.logger.log_metrics({'train/time': diff_seconds})


class ValidationDurationCallback(Callback):
    def on_validation_start(self, trainer, pl_module) -> None:
        self.start_time = time.monotonic()

    def on_validation_end(self, trainer, pl_module, unused=None):
        end_time = time.monotonic()
        diff = timedelta(seconds=end_time - self.start_time)
        diff_seconds = diff.total_seconds()
        trainer.logger.log_metrics({'val/time': diff_seconds})


class TestDurationCallback(Callback):
    def on_test_start(self, trainer, pl_module) -> None:
        self.start_time = time.monotonic()

    def on_test_end(self, trainer, pl_module, unused=None):
        end_time = time.monotonic()
        diff = timedelta(seconds=end_time - self.start_time)
        diff_seconds = diff.total_seconds()
        trainer.logger.log_metrics({'test/time': diff_seconds})
