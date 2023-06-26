import logging

import pandas as pd
from unidecode import unidecode

from src.data_preprocessing.wsd_disambiguate import DisambiguatorOnline
from src.settings import DATA_DIR

logging.basicConfig(level=logging.INFO)

# dataset = 'goemotions'
# split = 'test'
# language = 'EN'
#
# DATASET_DIR = DATA_DIR.joinpath(dataset)
#
# df = pd.read_csv(DATASET_DIR.joinpath(f'{split}.tsv'), sep='\t')
# print(f'Dataframe len: {len(df)}')
#
# texts = df['text'].values
# indices = [idx for idx in range(len(texts))]
#
# df = pd.DataFrame(zip(indices, texts), columns=['text_id', 'text'], dtype=str)
#
# disambiguator = DisambiguatorOnline(
#     df=df,
#     text_col='text',
#     id_col='text_id',
#     lang=language,
#     batch=5,
#     save_jsonl_path=DATASET_DIR.joinpath(f'wsd_amuse_{language}', f'{split}.amuse_wsd.jsonl'),
# )
# disambiguator.AMUSE_WSD_URL = 'http://127.0.0.1:12345/api/model'
# disambiguator.fetch_results()
# ----------------------------------------------------------------------------------------------------------------------
# dataset = 'klej_ar'
# split = 'train'
# language = 'PL'
#
# DATASET_DIR = DATA_DIR.joinpath(dataset)
#
# df = pd.read_csv(DATASET_DIR.joinpath(f'{split}.tsv'), sep='\t')
# print(f'Dataframe len: {len(df)}')
# texts = df['text'].values
# indices = [idx for idx in range(len(texts))]
#
# df = pd.DataFrame(zip(indices, texts), columns=['text_id', 'text'], dtype=str)
# disambiguator = DisambiguatorOnline(
#     df=df,
#     text_col='text',
#     id_col='text_id',
#     lang=language,
#     batch=1,
#     tmp_filename=f'__tmp_fetched_{dataset}_{split}',
#     save_jsonl_path=DATASET_DIR.joinpath(f'wsd_amuse_{language}', f'{split}.amuse_wsd.jsonl')
# )
# disambiguator.AMUSE_WSD_URL = 'http://127.0.0.1:12346/api/model'
# disambiguator.fetch_results()
# # disambiguator.jsonlines_to_json()
# ----------------------------------------------------------------------------------------------------------------------
# dataset = 'polemo2'
# split = 'train'
# language = 'PL'
# domain = 'all'
# mode = 'text'
#
# DATASET_DIR = DATA_DIR.joinpath(dataset, 'txt', f'{domain}_{mode}_{split}')

# records = []
# for filepath in DATASET_DIR.iterdir():
#     sample_id = int(filepath.stem)
#     text = filepath.read_text()
#     records.append({'text_id': sample_id, 'text': text})
#
# df = pd.DataFrame.from_records(records)
# df = df.sort_values(by=['text_id'])
# df = df.astype(str)
#
# disambiguator = DisambiguatorOnline(
#     df=df,
#     text_col='text',
#     id_col='text_id',
#     lang=language,
#     batch=1,
#     tmp_filename=f'__tmp_fetched_{dataset}_{split}',
#     save_jsonl_path=DATA_DIR.joinpath('polemo2', f'wsd_amuse_{language}', f'{split}.amuse_wsd.jsonl')
# )
# disambiguator.AMUSE_WSD_URL = 'http://127.0.0.1:12346/api/model'
# disambiguator.fetch_results()

# ----------------------------------------------------------------------------------------------------------------------
dataset = 'multiemo'
split = 'train'
language = 'EN'
domain = 'all'
mode = 'text'

DATASET_DIR = DATA_DIR.joinpath(dataset)
FILEPATH = DATASET_DIR.joinpath(f'{domain}.{mode}.{split}.txt')

with FILEPATH.open(mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = [line.rstrip('\n').split(' __label__') for line in lines]

records = []
for idx, (text, label) in enumerate(lines):
    records.append({'text_id': idx, 'text': unidecode(text)})

df = pd.DataFrame.from_records(records)
df = df.sort_values(by=['text_id'], ascending=True)
df = df.astype(str)

disambiguator = DisambiguatorOnline(
    df=df,
    text_col='text',
    id_col='text_id',
    lang=language,
    batch=1,
    tmp_filename=f'__tmp_fetched_{dataset}_{split}',
    save_jsonl_path=DATASET_DIR.joinpath(f'wsd_amuse_{language}', f'{split}.amuse_wsd.jsonl'),
)
disambiguator.AMUSE_WSD_URL = 'http://127.0.0.1:12345/api/model'
disambiguator.fetch_results()
disambiguator.jsonlines_to_json()
