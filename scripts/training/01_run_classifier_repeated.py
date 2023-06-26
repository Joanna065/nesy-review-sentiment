import logging
import subprocess
from dataclasses import dataclass

from smartparams import Smart

from src.settings import PARAMS_DIR

logger = logging.getLogger(__name__)


@dataclass
class Config:
    seed: int
    mode: str
    num_repeats: int
    train_sizes: dict[str, list[int]]


def main(smart: Smart[Config]):
    # setup ------------------------------------------------------------------------------------------------------------
    params = smart()

    with smart.lab.cache(path=PARAMS_DIR) as params_dir:
        path = params_dir.joinpath('params.yaml')
        params_path_smart = Smart()
        if params.mode:
            for mode in params.mode.split(','):
                params_path_smart.update_from(params_dir.joinpath(f'params.{mode}.yaml'))

        dataset_class = Smart.register.callable(
            params_path_smart.get('datamodule.dataset_cls.class')
        )
        train_sizes = params.train_sizes[dataset_class.NAME]
        print(f'All train sizes to run: {train_sizes}')

        for train_size in train_sizes:
            for i in range(params.num_repeats):
                print(f'--- Dataset: {dataset_class} ---')
                print(f'--- Mode: {params.mode} ---')
                print(f'--- Train size: {train_size} ---')
                print(f'--- Repeat nr: {i} ---')

                seed = params.seed + i
                command = [
                    'python',
                    '-m',
                    'scripts.training.01_run_classifier',
                    f'--path={path}',
                    f'seed={seed}',
                    f'datamodule.train_size={int(train_size)}',
                    f'.repeat={i}',
                ]
                if params.mode:
                    command.append(f'--mode={params.mode},{dataset_class.NAME}')

                print('Running:', ' '.join(command))
                subprocess.run(command, check=True)


if __name__ == '__main__':
    Smart(Config).run(
        function=main,
        path=PARAMS_DIR.joinpath('repeated.yaml'),
    )
