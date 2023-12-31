import pandas as pd
import wandb

from src.settings import WANDB_EXPORT_DIR

api = wandb.Api(timeout=20)

GROUP_NAME = 'amc'
PROJECT_NAME = 'neurosymbolics_batch_8'

runs = api.runs(f'{GROUP_NAME}/{PROJECT_NAME}')
records = [
    {
        **{
            k: v
            for k, v in {**run.config, **run.summary._json_dict}.items()
            if not k.startswith('_') and 'gradients' not in k
        },
        'name': run.name,
        'state': run.state,
    }
    for run in runs
]

runs_df = pd.DataFrame(records)
runs_df.to_csv(WANDB_EXPORT_DIR.joinpath(f'wandb_results_{PROJECT_NAME}.csv'))
