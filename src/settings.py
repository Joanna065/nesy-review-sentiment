from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
PARAMS_FILE = PROJECT_DIR.joinpath('params.yaml')

STORAGE_DIR = PROJECT_DIR / 'storage'
DATA_DIR = STORAGE_DIR / 'data'
EXPERIMENTS_DIR = STORAGE_DIR / 'experiments'
PARAMS_DIR = STORAGE_DIR / 'params'
FIGURE_DIR = STORAGE_DIR / 'figures'
SENTIMENT_RES_DIR = STORAGE_DIR / 'sent_emo_resources'
KEPLER_KE_DIR = STORAGE_DIR / 'kepler_KE'
SENTI_LARE_DIR = STORAGE_DIR / 'sentiLARE'
WANDB_EXPORT_DIR = STORAGE_DIR / 'wandb_export'
