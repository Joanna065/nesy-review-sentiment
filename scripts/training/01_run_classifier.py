import json
import logging
from dataclasses import dataclass

import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from smartparams import Smart

from src.callbacks.memory import ModelSize
from src.callbacks.time import TrainDurationCallback, TestDurationCallback
from src.callbacks.wandb import LogConfusionMatrix, WatchModel
from src.data.main_datamodule import MainDataModule
from src.models.baseline import BaselineModel
from src.settings import EXPERIMENTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class Config:
    seed: int
    datamodule: Smart[MainDataModule]
    model: Smart[BaselineModel]
    wandb_logger: Smart[WandbLogger]
    checkpoint: Smart[ModelCheckpoint]
    early_stopping: Smart[EarlyStopping]
    trainer: Smart[Trainer]


def main(smart: Smart[Config]):
    # setup ------------------------------------------------------------------------------------------------------------
    params = smart()
    seed_everything(params.seed, workers=True)

    # datamodule -------------------------------------------------------------------------------------------------------
    datamodule = params.datamodule()
    smart.set('data_name', datamodule.dataset_cls.NAME)
    smart.set('data_source', datamodule.dataset_cls.DATA_SOURCE)
    smart.set('data_lang', datamodule.dataset_cls.LANGUAGE)

    # model ------------------------------------------------------------------------------------------------------------
    model = params.model(
        num_labels=datamodule.dataset_cls.NUM_LABELS,
        multilabel=datamodule.dataset_cls.MULTILABEL,
    )

    # create experiment ------------------------------------------------------------------------------------------------
    save_dir = smart.lab.new(
        save_dir=EXPERIMENTS_DIR,
        version=f'{model.NAME}_{datamodule.dataset_cls.NAME}' + '_{Y}_{m}_{d}_{H}_{M}',
    )
    smart.lab.metadata(
        include_git=False,
        save_to=save_dir.joinpath('metadata.yaml'),
        config=smart.dict(),
    )

    # logger -----------------------------------------------------------------------------------------------------------
    logger.info("Initialize wandb logger...")
    wandb_logger = params.wandb_logger(
        name=save_dir.name,
        save_dir=str(save_dir),
    )
    wandb_logger.log_hyperparams(smart.dict(unpack=True))

    # callbacks --------------------------------------------------------------------------------------------------------
    checkpoint_dir = save_dir.joinpath('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    callbacks = [
        WatchModel(),
        ModelSize(),
        TrainDurationCallback(),
        TestDurationCallback(),
        # LogTrainingDynamics(save_dir=save_dir),
        LogConfusionMatrix(
            num_classes=datamodule.dataset_cls.NUM_LABELS,
            log_modes=('test',),
        ),
        LearningRateMonitor(logging_interval='step'),
        params.checkpoint(
            dirpath=checkpoint_dir,
        ),
        params.early_stopping(),
    ]

    # trainer ----------------------------------------------------------------------------------------------------------
    trainer = params.trainer(
        logger=wandb_logger,
        callbacks=callbacks,
    )

    logger.info("Starting training...")
    wandb.require(experiment="service")  # opt-in for lightning and DDP / multiprocessing

    # fitting ----------------------------------------------------------------------------------------------------------
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )

    # evaluation -------------------------------------------------------------------------------------------------------
    logger.info("Evaluating test split for best checkpoint model...")
    metrics, *_ = trainer.test(datamodule=datamodule, ckpt_path='best')

    result_dir = save_dir.joinpath('metrics')
    result_dir.mkdir(exist_ok=True, parents=True)
    with result_dir.joinpath('test_metrics.json').open('w') as file:
        json.dump(metrics, file, indent=2)


if __name__ == '__main__':
    Smart(Config).run(
        function=main,
        # path=PARAMS_DIR.joinpath('params.yaml'),
        # mode='baseline,dev',
    )
