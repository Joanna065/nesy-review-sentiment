import json
import logging

import pandas as pd
from tqdm import tqdm

from src.settings import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# English datasets
dataset = 'stanford_treebank'
split = 'train'
language = 'EN'

DATASET_DIR = DATA_DIR.joinpath(dataset)
WSD_FILEPATH = DATASET_DIR.joinpath(f'wsd_amuse_{language}', f'{dataset}.{split}.wsd.json')

df_data = pd.read_csv(DATASET_DIR.joinpath(f'{split}.tsv'), sep='\t')
labels = df_data.label.values

with WSD_FILEPATH.open(mode='r') as f:
    data = json.load(f)
    logger.info("Read %d dataset records", len(data))

for sample_id in tqdm(data.keys(), desc="Processing SentiWordnet annotations..."):
    sample_data = data[sample_id]
    sample_data['label'] = int(labels[int(sample_id)])

with WSD_FILEPATH.open(mode='w') as f:
    json.dump(data, f, indent=2)

# # GoEmotions - multilabel dataset
# dataset = 'goemotions'
# split = 'train'
# language = 'EN'
#
# DATASET_DIR = DATA_DIR.joinpath(dataset)
# WSD_FILEPATH = DATASET_DIR.joinpath(f'wsd_amuse_{language}', f'{dataset}.{split}.wsd.json')
#
# df_data = pd.read_csv(DATASET_DIR.joinpath(f'{split}.tsv'), sep='\t')
# labels = df_data.labels.values.tolist()
#
# labels = [label.split(',') for label in labels]
#
#
# with WSD_FILEPATH.open(mode='r') as f:
#     data = json.load(f)
#     logger.info("Read %d dataset records", len(data))
#
# for sample_id in tqdm(data.keys(), desc="Processing SentiWordnet annotations..."):
#     sample_data = data[sample_id]
#     sample_data['label'] = [int(label) for label in labels[int(sample_id)]]
#
# with WSD_FILEPATH.open(mode='w') as f:
#     json.dump(data, f, indent=2)


# # POLISH POLEMO 2.0
# dataset = 'polemo2'
# split = 'test'
# language = 'PL'
#
# domain = 'all'
# mode = 'text'
#
# DATASET_DIR = DATA_DIR.joinpath(dataset)
# with DATASET_DIR.joinpath('origin', f'{domain}.{mode}.{split}.txt').open() as f:
#     lines = f.readlines()
#
# texts = []
# labels = []
#
# for line in lines:
#     text, label = line.split('__label__')
#     labels.append(label.rstrip('\n'))
#     texts.append(text.strip())
#
# df_data = pd.DataFrame(list(zip(texts, labels)), columns=['text', 'label'])
# labels = df_data.label.values
#
# WSD_FILEPATH = DATASET_DIR.joinpath(f'wsd_amuse_{language}', f'{domain}_{mode}.{split}.wsd.json')
#
# with WSD_FILEPATH.open(mode='r') as f:
#     data = json.load(f)
#     logger.info("Read %d dataset records", len(data))
#
# for sample_id in tqdm(data.keys(), desc="Processing SentiWordnet annotations..."):
#     sample_data = data[sample_id]
#     sample_data['label'] = labels[int(sample_id)]
#
# with WSD_FILEPATH.open(mode='w') as f:
#     json.dump(data, f, indent=2)
